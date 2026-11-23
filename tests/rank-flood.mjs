import { combinedScore, pickMatches, cosine, l2norm, createScoreTracker, ACQUIRE } from '../src/ml/rank.js';
import { floodFromPoint, maskedCrop } from '../src/ml/flood.js';

let failed = 0;
function assert(cond, msg) {
  if (!cond) { failed++; console.error('FAIL:', msg); }
  else console.log('ok:', msg);
}

{
  const a = l2norm([1, 0, 0, 0]);
  const b = l2norm([1, 0, 0, 0]);
  const c = l2norm([0, 1, 0, 0]);
  assert(Math.abs(cosine(a, b) - 1) < 1e-5, 'identical vectors cosine = 1');
  assert(Math.abs(cosine(a, c)) < 1e-5, 'orthogonal vectors cosine = 0');
}

{
  const strong = combinedScore(0.92, 0.8);
  const mid = combinedScore(0.78, 0.6);
  const weak = combinedScore(0.48, 0.5);
  assert(strong > mid && mid > weak, `combined ranks pattern first (${strong.toFixed(2)} > ${mid.toFixed(2)} > ${weak.toFixed(2)})`);
  assert(strong >= ACQUIRE, 'a clear pattern+color match clears acquire');
  assert(weak < ACQUIRE, 'a weak pattern and middling color stays below acquire');
}

{
  const cands = [
    { id: 1, score: 0.88 },
    { id: 2, score: 0.81 },
    { id: 3, score: 0.79 },
    { id: 4, score: 0.77 },
    { id: 5, score: 0.40 },
  ];
  const top = pickMatches(cands);
  assert(top.length >= 1 && top.length <= 3, `picks 1–3 matches, got ${top.length}`);
  assert(top.every((c) => c.score >= ACQUIRE), 'every pick is above the acquire bar');
  assert(pickMatches([{ id: 9, score: 0.5 }]).length === 0, 'shows nothing when unsure');
}

{
  const tr = createScoreTracker({ acquire: 0.7, drop: 0.62, holdMs: 400, minHits: 2 });
  const now = 1000;
  let shown = tr.update([{ id: 1, score: 0.85, cosine: 0.9, color: 0.7 }], now);
  shown = tr.update([{ id: 1, score: 0.86, cosine: 0.9, color: 0.7 }], now + 80);
  assert(shown.some((s) => s.id === 1), 'tracker acquires after two good hits');
  // brief miss
  shown = tr.update([], now + 160);
  assert(shown.some((s) => s.id === 1), 'tracker holds through a brief miss');
  // long miss
  let last = shown;
  for (let t = now + 200; t <= now + 1200; t += 80) last = tr.update([], t);
  assert(!last.some((s) => s.id === 1), 'tracker drops after ~1s of misses');
}

{
  // Synthetic sock: beige linen + a red elongated blob.
  const W = 160, H = 220;
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < rgba.length; i += 4) { rgba[i] = 210; rgba[i+1] = 200; rgba[i+2] = 185; rgba[i+3] = 255; }
  const cx = 80, cy = 110, rw = 18, rh = 50;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const e = ((x - cx) / rw) ** 2 + ((y - cy) / rh) ** 2;
      if (e <= 1) {
        const i = (y * W + x) * 4;
        rgba[i] = 200; rgba[i+1] = 40; rgba[i+2] = 50;
      }
    }
  }
  const r = floodFromPoint(rgba, W, H, cx, cy);
  assert(r.ok, 'flood finds the synthetic sock');
  assert(r.area > 400 && r.area < 5000, `flood area is sock-sized (${r.area})`);
  const crop = maskedCrop(rgba, W, H, r, 4);
  assert(crop.w > 20 && crop.h > 40, 'masked crop has a sock-like aspect');
}

if (failed) { console.error(`\n${failed} failed`); process.exit(1); }
console.log('\nAll rank/flood checks passed');
