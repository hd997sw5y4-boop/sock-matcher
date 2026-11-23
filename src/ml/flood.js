/**
 * Tap-to-pick fallback when the segmentation model is not ready or fails.
 * Flood-fills from the tap in Lab, stopping at strong edges, then keeps the
 * component that contains the seed. No OpenCV, no model.
 */
import { rgbToLab } from '../cv/lab.js';

export function floodFromPoint(rgba, w, h, x, y, opts = {}) {
  x = Math.max(0, Math.min(w - 1, Math.round(x)));
  y = Math.max(0, Math.min(h - 1, Math.round(y)));
  const n = w * h;
  const L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const lab = rgbToLab(rgba[p], rgba[p + 1], rgba[p + 2]);
    L[i] = lab[0]; A[i] = lab[1]; B[i] = lab[2];
  }
  const grad = new Float32Array(n);
  for (let yy = 1; yy < h - 1; yy++) {
    for (let xx = 1; xx < w - 1; xx++) {
      const i = yy * w + xx;
      const dx = L[i + 1] - L[i - 1], dy = L[i + w] - L[i - w];
      grad[i] = Math.hypot(dx, dy) * 0.5;
    }
  }
  const seed = y * w + x;
  const sL = L[seed], sA = A[seed], sB = B[seed];
  const minArea = Math.max(40, n * (opts.minFrac ?? 0.002));
  const maxArea = Math.round(n * (opts.maxFrac ?? 0.18));
  const edgeCut = opts.edgeCut ?? 18;

  function run(thr) {
    const seen = new Uint8Array(n);
    const q = [seed];
    seen[seed] = 1;
    let meanL = sL, meanA = sA, meanB = sB, count = 1;
    for (let qi = 0; qi < q.length; qi++) {
      const i = q[qi];
      const cx = i % w, cy = (i / w) | 0;
      for (const j of [i - 1, i + 1, i - w, i + w]) {
        if (j < 0 || j >= n || seen[j]) continue;
        const xx = j % w, yy = (j / w) | 0;
        if (Math.abs(xx - cx) + Math.abs(yy - cy) !== 1) continue;
        seen[j] = 1;
        if (grad[j] > edgeCut && colorDist(L[j], A[j], B[j], sL, sA, sB) > thr * 0.55) continue;
        const dSeed = colorDist(L[j], A[j], B[j], sL, sA, sB);
        const dMean = colorDist(L[j], A[j], B[j], meanL, meanA, meanB);
        if (Math.min(dSeed, dMean) > thr) continue;
        q.push(j);
        const k = q.length;
        meanL += (L[j] - meanL) / k;
        meanA += (A[j] - meanA) / k;
        meanB += (B[j] - meanB) / k;
        if (q.length >= maxArea) return q;
      }
    }
    return q;
  }

  let best = null;
  for (const thr of [12, 16, 20, 26, 32]) {
    const q = run(thr);
    if (q.length >= minArea && q.length < maxArea) { best = q; break; }
    if (q.length >= minArea && (!best || Math.abs(q.length - minArea * 4) < Math.abs(best.length - minArea * 4))) best = q;
  }
  if (!best || best.length < minArea * 0.5) {
    return { ok: false, mask: new Uint8Array(n), area: 0, box: [x, y, x, y], cx: x, cy: y };
  }

  let mask = new Uint8Array(n);
  for (const i of best) mask[i] = 1;
  mask = closeMask(mask, w, h, 1);
  // Keep only the component that still contains the seed.
  const keep = new Uint8Array(n);
  if (mask[seed]) {
    const q = [seed]; keep[seed] = 1;
    for (let qi = 0; qi < q.length; qi++) {
      const i = q[qi];
      for (const j of [i - 1, i + 1, i - w, i + w]) {
        if (j < 0 || j >= n || keep[j] || !mask[j]) continue;
        keep[j] = 1; q.push(j);
      }
    }
    mask = keep;
  }
  return maskToRegion(mask, w, h);
}

function colorDist(l1, a1, b1, l2, a2, b2) {
  const dl = (l1 - l2) * 0.5, da = a1 - a2, db = b1 - b2;
  return Math.sqrt(dl * dl + da * da + db * db);
}

function closeMask(mask, w, h, r) {
  const n = w * h;
  const tmp = new Uint8Array(n);
  dilate(mask, tmp, w, h, r);
  const out = new Uint8Array(n);
  erode(tmp, out, w, h, r);
  return out;
}

function dilate(src, dst, w, h, r) {
  dst.fill(0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!src[y * w + x]) continue;
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= w) continue;
          dst[yy * w + xx] = 1;
        }
      }
    }
  }
}

function erode(src, dst, w, h, r) {
  dst.fill(0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!src[y * w + x]) continue;
      let ok = true;
      for (let dy = -r; dy <= r && ok; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= h) { ok = false; break; }
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w || !src[yy * w + xx]) { ok = false; break; }
        }
      }
      if (ok) dst[y * w + x] = 1;
    }
  }
}

export function maskToRegion(mask, w, h) {
  let minX = w, minY = h, maxX = 0, maxY = 0, area = 0, sx = 0, sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      area++; sx += x; sy += y;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  if (!area) return { ok: false, mask, area: 0, box: [0, 0, 0, 0], cx: 0, cy: 0 };
  return {
    ok: area >= 24,
    mask,
    area,
    box: [minX, minY, maxX, maxY],
    cx: sx / area,
    cy: sy / area,
  };
}

/** Paint a masked sock crop: keep sock pixels, fill the rest with a neutral cloth color. */
export function maskedCrop(rgba, w, h, region, pad = 6) {
  const [x0, y0, x1, y1] = region.box;
  const left = Math.max(0, x0 - pad), top = Math.max(0, y0 - pad);
  const right = Math.min(w - 1, x1 + pad), bottom = Math.min(h - 1, y1 + pad);
  const cw = right - left + 1, ch = bottom - top + 1;
  const out = new Uint8ClampedArray(cw * ch * 4);
  // Neutral fill — a mid linen, so the model sees the sock shape, not the floor.
  const fill = [210, 200, 188];
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const i = ((y * cw) + x) * 4;
      const sx = left + x, sy = top + y;
      const on = region.mask[sy * w + sx];
      const p = (sy * w + sx) * 4;
      if (on) {
        out[i] = rgba[p]; out[i + 1] = rgba[p + 1]; out[i + 2] = rgba[p + 2]; out[i + 3] = 255;
      } else {
        out[i] = fill[0]; out[i + 1] = fill[1]; out[i + 2] = fill[2]; out[i + 3] = 255;
      }
    }
  }
  return { rgba: out, w: cw, h: ch, left, top };
}

/** Crop a track box with padding, no mask (used for live proposals). */
export function boxCrop(rgba, w, h, box, pad = 4) {
  const x0 = Math.max(0, Math.floor(box[0] - pad));
  const y0 = Math.max(0, Math.floor(box[1] - pad));
  const x1 = Math.min(w - 1, Math.ceil(box[2] + pad));
  const y1 = Math.min(h - 1, Math.ceil(box[3] + pad));
  const cw = Math.max(1, x1 - x0 + 1), ch = Math.max(1, y1 - y0 + 1);
  const out = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    const src = ((y0 + y) * w + x0) * 4;
    out.set(rgba.subarray(src, src + cw * 4), y * cw * 4);
  }
  return { rgba: out, w: cw, h: ch, left: x0, top: y0 };
}

/** Rasterize a centroid-relative outline (flat [dx,dy,...]) into a region. */
export function outlineToRegion(outline, cx, cy, w, h) {
  if (!outline || outline.length < 6) return null;
  const mask = new Uint8Array(w * h);
  const pts = [];
  let minX = w, minY = h, maxX = 0, maxY = 0;
  for (let i = 0; i < outline.length; i += 2) {
    const x = Math.round(cx + outline[i]);
    const y = Math.round(cy + outline[i + 1]);
    pts.push([x, y]);
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  minX = Math.max(0, minX); minY = Math.max(0, minY);
  maxX = Math.min(w - 1, maxX); maxY = Math.min(h - 1, maxY);
  // Even-odd scan fill.
  for (let y = minY; y <= maxY; y++) {
    const xs = [];
    for (let i = 0, n = pts.length; i < n; i++) {
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[(i + 1) % n];
      if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
        xs.push(x1 + (y - y1) * (x2 - x1) / (y2 - y1 || 1));
      }
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const a = Math.max(minX, Math.round(xs[k]));
      const b = Math.min(maxX, Math.round(xs[k + 1]));
      for (let x = a; x <= b; x++) mask[y * w + x] = 1;
    }
  }
  return maskToRegion(mask, w, h);
}
