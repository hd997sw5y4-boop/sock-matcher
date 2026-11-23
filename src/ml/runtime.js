/**
 * Load and run the on-device pattern models.
 * - DINOv2-small (Apache-2.0, Meta) for visual embeddings (~24 MB q8)
 * - SlimSAM-77 (Apache-2.0) for tap-to-segment (~14 MB q8)
 * Works in a browser worker (WebGPU, WASM fallback) or in Node.
 */
import { floodFromPoint, maskedCrop, boxCrop, maskToRegion, outlineToRegion } from './flood.js';
import { cosine, l2norm, combinedScore, pickMatches } from './rank.js';

export const DINO_ID = 'Xenova/dinov2-small';
export const DTYPE = 'q4'; // ~15 MB, suitable for iPhone Safari

export function isIOS() {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent || '';
  return /iP(hone|ad|od)/.test(ua) || (navigator.platform === 'MacIntel' && (navigator.maxTouchPoints || 0) > 1);
}

export function configureEnv(env, { cacheDir, localModelPath } = {}) {
  if (cacheDir) {
    env.cacheDir = cacheDir;
    env.useFSCache = true;
  }
  if (typeof navigator !== 'undefined') {
    env.allowLocalModels = true;
    env.allowRemoteModels = true;
    env.localModelPath = localModelPath || '/models/';
    const wasm = env.backends?.onnx?.wasm;
    if (wasm) {
      // Same-origin WASM, one thread, no proxy. Never start a WebGPU backend
      // alongside WASM — that is a heat and memory problem on phones.
      wasm.proxy = false;
      wasm.numThreads = 1;
      wasm.wasmPaths = '/vendor/onnx/';
    }
  }
}

function onProgress(cb, phase, base, span) {
  return (info) => {
    if (!cb) return;
    let frac = 0;
    if (typeof info?.progress === 'number') frac = info.progress / 100;
    else if (info?.loaded && info?.total) frac = info.loaded / info.total;
    const file = String(info?.file || info?.name || '');
    const status = info?.status || phase;
    cb({ phase, status, file, progress: Math.round((base + frac * span) * 100) });
  };
}

/** WASM only. Do not attempt WebGPU — and never run both backends. */
export async function loadExtractor(tf, { progress } = {}) {
  return tf.pipeline(
    'image-feature-extraction',
    DINO_ID,
    { dtype: DTYPE, device: 'wasm', progress_callback: onProgress(progress, 'dino', 0, 1) },
  );
}

export function rgbaToRaw(tf, rgba, w, h) {
  // Drop alpha — RawImage wants RGB or RGBA; RGB is slightly smaller to process.
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2];
  }
  return new tf.RawImage(rgb, w, h, 3);
}

export async function embedRaw(extractor, raw) {
  // DINOv2-small has no pooler; use a blend of the CLS token and mean of patch tokens.
  const out = await extractor(raw);
  const data = out.data || out;
  const dims = out.dims || [1, data.length];
  if (dims.length === 1 || (dims.length === 2 && dims[0] === 1) || (dims.at(-1) === data.length)) {
    return l2norm(data instanceof Float32Array ? data : new Float32Array(data));
  }
  const t = dims.length === 3 ? dims[1] : dims[0];
  const c = dims.length === 3 ? dims[2] : dims[1];
  const cls = new Float32Array(c);
  const mean = new Float32Array(c);
  for (let j = 0; j < c; j++) cls[j] = data[j];
  const patches = Math.max(1, t - 1);
  for (let i = 1; i < t; i++) {
    const off = i * c;
    for (let j = 0; j < c; j++) mean[j] += data[off + j];
  }
  const mixed = new Float32Array(c);
  for (let j = 0; j < c; j++) mixed[j] = 0.5 * cls[j] + 0.5 * (mean[j] / patches);
  return l2norm(mixed);
}

export async function embedCrop(tf, extractor, cropRgba, cw, ch) {
  if (cw < 4 || ch < 4) return null;
  const raw = rgbaToRaw(tf, cropRgba, cw, ch);
  return embedRaw(extractor, raw);
}

/**
 * Point-prompt SlimSAM. Image is downscaled so a phone can encode it quickly.
 * Returns a binary mask in the original frame size, or null on failure.
 */
export async function samPick(tf, sam, rgba, w, h, point) {
  if (!sam) return null;
  try {
    const long = Math.max(w, h);
    const scale = long > 512 ? 512 / long : 1;
    const sw = Math.max(16, Math.round(w * scale)), sh = Math.max(16, Math.round(h * scale));
    let src = rgba, tw = w, th = h;
    if (scale !== 1) {
      src = downsample(rgba, w, h, sw, sh);
      tw = sw; th = sh;
    }
    const raw = rgbaToRaw(tf, src, tw, th);
    const input_points = [[[point.x * (tw / w), point.y * (th / h)]]];
    const inputs = await sam.processor(raw, { input_points });
    const outputs = await sam.model(inputs);
    const masks = await sam.processor.post_process_masks(
      outputs.pred_masks,
      inputs.original_sizes,
      inputs.reshaped_input_sizes,
    );
    const scores = outputs.iou_scores.data || outputs.iou_scores;
    let best = 0, bestS = -1;
    const n = scores.length ?? 0;
    for (let i = 0; i < n; i++) if (scores[i] > bestS) { bestS = scores[i]; best = i; }
    if (bestS < 0.45) return null;
    const tensor = Array.isArray(masks) ? masks[0] : masks;
    // dims: [1, 3, H, W] or a Tensor list
    const mask = extractMask(tensor, best, tw, th);
    if (!mask) return null;
    const full = scale !== 1 ? upsampleMask(mask, tw, th, w, h) : mask;
    const region = maskToRegion(full, w, h);
    if (!region.ok) return null;
    // The tap should land in (or very near) the mask.
    const sid = Math.round(point.y) * w + Math.round(point.x);
    const near = full[sid] || neighborsOn(full, w, h, Math.round(point.x), Math.round(point.y), 4);
    if (!near) return null;
    region.score = bestS;
    region.method = 'sam';
    return region;
  } catch (err) {
    console.warn('SAM pick failed', err);
    return null;
  }
}

function neighborsOn(mask, w, h, x, y, r) {
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      if (mask[yy * w + xx]) return true;
    }
  }
  return false;
}

function extractMask(tensor, index, w, h) {
  // Support Tensor with dims [1, 3, H, W] or [3, H, W] and a data Uint8/bool array.
  if (!tensor) return null;
  const t = tensor[index] && tensor[index].data ? tensor[index] : tensor;
  const dims = t.dims || t.size && w && h ? t.dims : null;
  const data = t.data || t;
  if (!data) return null;
  let mh, mw, stride, offset = 0;
  if (dims && dims.length === 4) {
    // [1, C, H, W]
    mh = dims[2]; mw = dims[3];
    stride = mh * mw;
    offset = index * stride;
  } else if (dims && dims.length === 3) {
    mh = dims[1]; mw = dims[2];
    stride = mh * mw;
    offset = index * stride;
  } else if (data.length === w * h) {
    mh = h; mw = w; stride = w * h; offset = 0;
  } else if (data.length >= (index + 1) * h * w) {
    mh = h; mw = w; stride = w * h; offset = index * stride;
  } else {
    return null;
  }
  const out = new Uint8Array(w * h);
  if (mw === w && mh === h) {
    for (let i = 0; i < stride; i++) out[i] = data[offset + i] ? 1 : 0;
    return out;
  }
  // Resize mask to the SAM input size first if needed, caller may upsample to frame.
  const tmp = new Uint8Array(mw * mh);
  for (let i = 0; i < tmp.length; i++) tmp[i] = data[offset + i] ? 1 : 0;
  return upsampleMask(tmp, mw, mh, w, h);
}

function downsample(rgba, w, h, nw, nh) {
  const out = new Uint8ClampedArray(nw * nh * 4);
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(h - 1, Math.round(y * h / nh));
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(w - 1, Math.round(x * w / nw));
      const si = (sy * w + sx) * 4, di = (y * nw + x) * 4;
      out[di] = rgba[si]; out[di + 1] = rgba[si + 1]; out[di + 2] = rgba[si + 2]; out[di + 3] = 255;
    }
  }
  return out;
}

function upsampleMask(src, sw, sh, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(sh - 1, Math.round(y * sh / h));
    for (let x = 0; x < w; x++) {
      const sx = Math.min(sw - 1, Math.round(x * sw / w));
      out[y * w + x] = src[sy * sw + sx] ? 1 : 0;
    }
  }
  return out;
}

export async function pickSock(tf, { extractor, sam }, rgba, w, h, point, box) {
  let region = await samPick(tf, sam, rgba, w, h, point);
  if (!region || !region.ok) {
    const seed = box
      ? { x: (box.x0 + box.x1) / 2, y: (box.y0 + box.y1) / 2 }
      : point;
    region = floodFromPoint(rgba, w, h, seed.x, seed.y);
    if (region.ok) region.method = 'flood';
  }
  if (!region.ok) return { ok: false };
  const crop = maskedCrop(rgba, w, h, region, 8);
  let embedding = null;
  if (extractor) {
    try { embedding = await embedCrop(tf, extractor, crop.rgba, crop.w, crop.h); }
    catch (err) { console.warn('embed ref failed', err); }
  }
  return { ok: true, ...region, embedding, crop };
}

/**
 * Score live proposals against a reference embedding.
 * `cache` is a Map of `${id}:${dv}` -> Float32Array.
 */
export const MAX_EMBED_CROPS = 6;

export async function scoreTracks(tf, extractor, rgba, w, h, tracks, refEmb, cache, { mode = 'scan', maxCrops = MAX_EMBED_CROPS } = {}) {
  const results = [];
  const ranked = (tracks || []).filter((t) => t && t.box)
    .slice()
    .sort((a, b) => (b.colorScore ?? 0) - (a.colorScore ?? 0));
  let newEmbeds = 0;
  for (const t of ranked) {
    const key = `${t.id}:${t.dv ?? 0}`;
    let emb = cache.get(key);
    if (!emb && extractor) {
      if (newEmbeds >= maxCrops) continue;
      newEmbeds++;
      const region = (t.outline && t.cx != null)
        ? outlineToRegion(t.outline, t.cx, t.cy, w, h)
        : null;
      const crop = (region && region.ok)
        ? maskedCrop(rgba, w, h, region, 6)
        : boxCrop(rgba, w, h, t.box, 4);
      try { emb = await embedCrop(tf, extractor, crop.rgba, crop.w, crop.h); }
      catch { emb = null; }
      if (emb) cache.set(key, emb);
    }
    if (!emb) continue;
    const cos = cosine(refEmb, emb);
    const color = t.colorScore ?? 0;
    results.push({
      id: t.id,
      cosine: cos,
      color,
      score: combinedScore(cos, color),
      box: t.box,
    });
  }
  if (mode === 'pairs') {
    const pairs = [];
    for (let i = 0; i < tracks.length; i++) {
      const a = tracks[i];
      const ea = cache.get(`${a.id}:${a.dv ?? 0}`);
      if (!ea) continue;
      for (let j = i + 1; j < tracks.length; j++) {
        const b = tracks[j];
        const eb = cache.get(`${b.id}:${b.dv ?? 0}`);
        if (!eb) continue;
        const cos = cosine(ea, eb);
        const color = 0; // color is filled in by the caller if it has classical sim
        pairs.push({
          id: `${a.id}:${b.id}`,
          a: a.id,
          b: b.id,
          cosine: cos,
          color,
          score: combinedScore(cos, color),
        });
      }
    }
    return { matches: [], pairs, scored: results };
  }
  return { matches: results, pairs: [], scored: results };
}

export { cosine, l2norm, combinedScore, pickMatches, floodFromPoint, maskedCrop, boxCrop, outlineToRegion };
