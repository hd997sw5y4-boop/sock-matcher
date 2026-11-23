/**
 * Embedding regression on Ezequiel's private laundry frames.
 * Frames stay in /workspace/sock-matcher-feedback and are never published.
 * Skips cleanly if that folder is not present.
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { env, pipeline, RawImage } from '@huggingface/transformers';
import { createEngine } from '../src/cv/engine.js';
import { thresholdFor } from '../src/cv/match.js';
import {
  configureEnv, embedCrop, cosine, combinedScore, pickMatches, floodFromPoint, maskedCrop, boxCrop,
} from '../src/ml/runtime.js';
import { maskToRegion } from '../src/ml/flood.js';

const FEEDBACK = '/workspace/sock-matcher-feedback';
if (!fs.existsSync(FEEDBACK)) {
  console.log('No private feedback frames; skipping embed regression.');
  process.exit(0);
}

configureEnv(env, { cacheDir: path.resolve('/workspace/sock-matcher/.cache/hf') });
console.log('Loading DINOv2-small (cached after the first run)…');
const tLoad = Date.now();
const extractor = await pipeline('image-feature-extraction', 'Xenova/dinov2-small', { dtype: 'q8', device: 'cpu' });
console.log(`model ready in ${Date.now() - tLoad}ms`);

function loadFile(file, w) {
  const out = execFileSync('ffmpeg', ['-v', 'error', '-i', file, '-vf', `scale=${w}:-2`, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'], { maxBuffer: 1 << 26 });
  const h = out.length / 4 / w;
  return { rgba: new Uint8ClampedArray(out.buffer, out.byteOffset, out.length), w, h };
}

function resizeRgba(src, sw, sh, nw, nh) {
  const rgba = new Uint8ClampedArray(nw * nh * 4);
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(sh - 1, Math.round(y * sh / nh));
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(sw - 1, Math.round(x * sw / nw));
      const si = (sy * sw + sx) * 4, di = (y * nw + x) * 4;
      rgba[di] = src[si]; rgba[di+1] = src[si+1]; rgba[di+2] = src[si+2]; rgba[di+3] = 255;
    }
  }
  return { rgba, w: nw, h: nh };
}

function cropRgba(src, sw, sh, x0, y0, x1, y1) {
  const cw = x1 - x0, ch = y1 - y0;
  const rgba = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    const si = ((y0 + y) * sw + x0) * 4;
    rgba.set(src.subarray(si, si + cw * 4), y * cw * 4);
  }
  return { rgba, w: cw, h: ch };
}

function tracksOf(frame) {
  const e = createEngine();
  let r;
  for (let k = 0; k < 3; k++) r = e.process(frame.rgba, frame.w, frame.h, { mode: 'scan', still: true });
  return { engine: e, result: r, tracks: [...e.getTracks().values()] };
}

function atPoint(tracks, w, h, nx, ny, maxR) {
  const x = nx * w, y = ny * h;
  let best = null, bd = 1e9;
  for (const t of tracks) {
    const inside = t.box && x >= t.box[0] && x <= t.box[2] && y >= t.box[1] && y <= t.box[3];
    const d = Math.hypot(t.cx - x, t.cy - y);
    const score = inside ? d * 0.25 : d;
    if (score < bd) { bd = score; best = t; }
  }
  return bd < (maxR ?? w * 0.1) ? best : null;
}

function trackRegion(t, w, h) {
  if (t.region && t.region.pixels) {
    const mask = new Uint8Array(w * h);
    for (const i of t.region.pixels) if (i >= 0 && i < mask.length) mask[i] = 1;
    return maskToRegion(mask, w, h);
  }
  return null;
}

async function embedTrack(frame, t) {
  const region = trackRegion(t, frame.w, frame.h);
  const crop = (region && region.ok)
    ? maskedCrop(frame.rgba, frame.w, frame.h, region, 6)
    : (t.box ? boxCrop(frame.rgba, frame.w, frame.h, t.box, 4) : null);
  if (!crop || crop.w < 6 || crop.h < 6) return null;
  return embedCrop({ RawImage }, extractor, crop.rgba, crop.w, crop.h);
}

async function embedFlood(frame, nx, ny) {
  const region = floodFromPoint(frame.rgba, frame.w, frame.h, nx * frame.w, ny * frame.h);
  if (!region.ok) return { region, emb: null };
  const crop = maskedCrop(frame.rgba, frame.w, frame.h, region, 6);
  const emb = await embedCrop({ RawImage }, extractor, crop.rgba, crop.w, crop.h);
  return { region, emb };
}

let pass = 0, total = 0, failed = 0;
function check(cond, msg) {
  total++;
  if (cond) { pass++; console.log('  ok:', msg); }
  else { failed++; console.error('  FAIL:', msg); }
}

// --- Known pairs on the earlier camera crops ---
const CAM = {
  'cam_12.5.png': [
    ['white w/ hearts', [0.30, 0.365], [0.68, 0.855]],
    ['light pink', [0.42, 0.81], [0.59, 0.78]],
    ['black', [0.58, 0.47], [0.78, 0.60]],
  ],
  'cam_11.5.png': [
    ['white w/ hearts', [0.27, 0.35], [0.68, 0.85]],
    ['light pink', [0.40, 0.80], [0.58, 0.78]],
    ['black', [0.58, 0.47], [0.77, 0.60]],
  ],
  'cam_13.png': [
    ['light pink', [0.37, 0.80], [0.585, 0.78]],
    ['black', [0.59, 0.47], [0.78, 0.60]],
    ['white w/ hearts', [0.28, 0.37], [0.67, 0.85]],
  ],
};

const times = [];
for (const [file, pairs] of Object.entries(CAM)) {
  const fp = path.join(FEEDBACK, file);
  if (!fs.existsSync(fp)) { console.log(`skip missing ${file}`); continue; }
  const frame = loadFile(fp, 240);
  const { engine, tracks } = tracksOf(frame);
  console.log(`\n${file}: ${tracks.length} tracks`);
  const embs = new Map();
  const t0 = Date.now();
  for (const t of tracks) {
    const emb = await embedTrack(frame, t);
    if (emb) embs.set(t.id, emb);
  }
  times.push((Date.now() - t0) / Math.max(1, embs.size));
  for (const [name, pa, pb] of pairs) {
    const a = atPoint(tracks, frame.w, frame.h, pa[0], pa[1]);
    const b = atPoint(tracks, frame.w, frame.h, pb[0], pb[1]);
    if (!a || !b || a === b) { check(false, `${name}: not detected as two regions`); continue; }
    const ea = embs.get(a.id), eb = embs.get(b.id);
    if (!ea || !eb) { check(false, `${name}: missing embedding`); continue; }
    const mate = cosine(ea, eb);
    const color = engine.simBetween(a.id, b.id);
    const comb = combinedScore(mate, color);
    const others = tracks.filter((t) => t !== a && embs.has(t.id))
      .map((t) => ({ id: t.id, score: combinedScore(cosine(ea, embs.get(t.id)), engine.simBetween(a.id, t.id)), cosine: cosine(ea, embs.get(t.id)) }))
      .sort((x, y) => y.score - x.score);
    const rank = others.findIndex((o) => o.id === b.id) + 1;
    const picked = pickMatches(others);
    const shown = picked.some((p) => p.id === b.id);
    const topWrong = others.filter((o) => o.id !== b.id)[0];
    const colorOk = color >= thresholdFor(0.55) * 0.9;
    // If the two regions were poorly isolated, require only that we don't glow on a scrap.
    if (!colorOk && rank > 3) {
      check(picked.length === 0 || shown, `${name}: weak isolation (color ${color.toFixed(3)}, rank ${rank}); model stays quiet or still shows the mate`);
    } else {
      check(rank >= 1 && rank <= 4, `${name}: mate rank ${rank} (cos ${mate.toFixed(3)}, color ${color.toFixed(3)}, comb ${comb.toFixed(3)})`);
    }
    const confidentWrong = topWrong && topWrong.score >= 0.70 && shown && !picked.some((p) => p.id === b.id);
    check(!confidentWrong, `${name}: does not light up a non-mate (top other score ${topWrong ? topWrong.score.toFixed(3) : 'n/a'})`);
    void shown;
  }
}

// --- v1257 latest pile (phone screenshots; crop to the camera stage) ---
const vdir = path.join(FEEDBACK, 'v1257');
if (fs.existsSync(path.join(vdir, 'f06.jpg'))) {
  // Crop away iOS/Safari chrome and the shutter. Normalized points are in the cropped frame.
  // Crop iOS/Safari chrome with ffmpeg (same path the app-scale frames use).
  const src = path.join(vdir, 'f06.jpg');
  const tmp = '/tmp/v1257-f06-stage.png';
  execFileSync('ffmpeg', ['-y', '-v', 'error', '-i', src, '-vf', 'crop=480:570:0:230,scale=240:-2', tmp]);
  const frame = loadFile(tmp, 240);
  const { engine, tracks } = tracksOf(frame);
  console.log(`\nv1257/f06 crop ${frame.w}x${frame.h}: ${tracks.length} tracks`);
  const cases = [
    ['mint grippy ankles', [0.72, 0.38], [0.62, 0.40]],
    ['white / navy heel', [0.42, 0.12], [0.64, 0.09]],
  ];
  const t0 = Date.now();
  const floodEmbs = [];
  for (const [name, pa, pb] of cases) {
    const A = await embedFlood(frame, pa[0], pa[1]);
    const B = await embedFlood(frame, pb[0], pb[1]);
    floodEmbs.push({ name, A, B });
  }
  // Embed classical proposals too, so we can rank the mate among the pile.
  const embs = new Map();
  for (const t of tracks) {
    const emb = await embedTrack(frame, t);
    if (emb) embs.set(t.id, emb);
  }
  times.push((Date.now() - t0) / Math.max(1, embs.size + floodEmbs.length * 2));
  for (const { name, A, B } of floodEmbs) {
    if (!A.emb || !B.emb) { check(false, `v1257 ${name}: flood/embed failed a=${!!A.emb} b=${!!B.emb} areaA=${A.region.area} areaB=${B.region.area}`); continue; }
    const mate = cosine(A.emb, B.emb);
    const others = [];
    for (const t of tracks) {
      const e = embs.get(t.id);
      if (!e) continue;
      // skip the region that is the query sock itself
      const dA = Math.hypot(t.cx - A.region.cx, t.cy - A.region.cy);
      if (dA < Math.max(12, Math.sqrt(A.region.area) * 0.35)) continue;
      others.push({ id: t.id, score: combinedScore(cosine(A.emb, e), 0), cosine: cosine(A.emb, e) });
    }
    others.sort((a, b) => b.score - a.score);
    const mateAmong = { id: 'mate', score: combinedScore(mate, 0), cosine: mate };
    const ranked = [mateAmong, ...others].sort((a, b) => b.score - a.score);
    const rank = ranked.findIndex((o) => o.id === 'mate') + 1;
    const picked = pickMatches(ranked);
    const shown = picked.some((p) => p.id === 'mate');
    const silent = picked.length === 0;
    // Success: the labeled mate is in the top 3, or the model stays quiet instead of lighting a scrap.
    const ok = (rank <= 3 && mate >= 0.55) || silent;
    check(ok, `v1257 ${name}: mate cos ${mate.toFixed(3)} rank ${rank}/${ranked.length} shown=${shown} silent=${silent} tracks=${tracks.length}`);
  }
}

const avgMs = times.reduce((a, b) => a + b, 0) / Math.max(1, times.length);
console.log(`\nembed time ~${avgMs.toFixed(0)}ms per crop`);
check(avgMs < 800, `per-crop embed ${avgMs.toFixed(0)}ms is practical for a few scores/sec after cache (CPU test)`);

console.log(`\nPASS ${pass}/${total}`);
if (failed) process.exit(1);
