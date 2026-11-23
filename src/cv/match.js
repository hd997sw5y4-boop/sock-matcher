/** Similarity between two sock descriptors, 0..1. */

function bhatt(p, q) {
  let s = 0;
  for (let i = 0; i < p.length; i++) s += Math.sqrt(p[i] * q[i]);
  return Math.min(1, s);
}

function cdist(l1, a1, b1, l2, a2, b2) {
  const dl = (l1 - l2) * 0.55, da = a1 - a2, db = b1 - b2;
  return Math.sqrt(dl * dl + da * da + db * db);
}

function palDist(p, q) {
  const one = (x, y) => {
    let s = 0, tw = 0;
    for (const c of x) {
      let best = 1e9;
      for (const d of y) best = Math.min(best, cdist(c.L, c.a, c.b, d.L, d.a, d.b));
      s += best * c.w; tw += c.w;
    }
    return s / (tw || 1);
  };
  return 0.5 * (one(p, q) + one(q, p));
}

function bandDist(x, y) {
  let f = 0, r = 0;
  const n = x.length / 3;
  for (let i = 0; i < n; i++) {
    const j = n - 1 - i;
    f += cdist(x[i * 3], x[i * 3 + 1], x[i * 3 + 2], y[i * 3], y[i * 3 + 1], y[i * 3 + 2]);
    r += cdist(x[i * 3], x[i * 3 + 1], x[i * 3 + 2], y[j * 3], y[j * 3 + 1], y[j * 3 + 2]);
  }
  return Math.min(f, r) / n;
}

/** Best of aligned cuff-to-toe bands and an unordered bag, so folds still compare. */
function bandDistFold(x, y) {
  if (!x || !y || x.length !== y.length) return 99;
  const aligned = bandDist(x, y);
  const n = x.length / 3;
  const px = [], py = [];
  for (let i = 0; i < n; i++) {
    px.push({ L: x[i * 3], a: x[i * 3 + 1], b: x[i * 3 + 2], w: 1 / n });
    py.push({ L: y[i * 3], a: y[i * 3 + 1], b: y[i * 3 + 2], w: 1 / n });
  }
  return Math.min(aligned, palDist(px, py));
}

/** Mean Lab distance of a 3x3 axis-aligned print grid, best of 180° flip. */
function gridDist(x, y) {
  if (!x || !y || x.length !== y.length) return 0;
  const n = x.length / 3;
  const fwd = [], rot = [];
  for (let i = 0; i < n; i++) {
    const j = n - 1 - i;
    fwd.push(cdist(x[i * 3], x[i * 3 + 1], x[i * 3 + 2], y[i * 3], y[i * 3 + 1], y[i * 3 + 2]));
    rot.push(cdist(x[i * 3], x[i * 3 + 1], x[i * 3 + 2], y[j * 3], y[j * 3 + 1], y[j * 3 + 2]));
  }
  const med = (arr) => { const a = arr.slice().sort((p, q) => p - q); return a[a.length >> 1]; };
  return Math.min(med(fwd), med(rot));
}

/**
 * Scan-mode score: appearance only. Color histograms, palette, stripe/pattern
 * texture. No size, area, aspect, length, or shape/contour terms — a bunched
 * or smaller/larger/rotated mate can still match.
 */
export function appearanceSimilarity(a, b) {
  if (!a || !b) return 0;
  const sAB = bhatt(a.ab, b.ab);
  const sL = bhatt(a.lh, b.lh);
  const sPal = Math.exp(-palDist(a.pal || [], b.pal || []) / 14);
  const sMean = Math.exp(-cdist(a.mean[0], a.mean[1], a.mean[2], b.mean[0], b.mean[1], b.mean[2]) / 10);
  const sBand = Math.exp(-bandDistFold(a.bands, b.bands) / 16);
  const sOri = a.ori && b.ori ? bhatt(a.ori, b.ori) : 1;
  const sEdge = Math.exp(-Math.abs((a.edge || 0) - (b.edge || 0)) / 0.09);
  const sStripe = Math.exp(-Math.abs((a.stripes || 0) - (b.stripes || 0)) / 3);
  const tex = 0.4 * sEdge + 0.35 * sOri + 0.25 * sStripe;
  // Missing grid (older saved scans) must not change the score.
  const sGrid = (a.grid && b.grid) ? Math.exp(-gridDist(a.grid, b.grid) / 14) : 1;
  return Math.exp(
    0.20 * Math.log(sAB + 1e-4)
    + 0.16 * Math.log(sPal + 1e-4)
    + 0.12 * Math.log(sMean + 1e-4)
    + 0.12 * Math.log(sBand + 1e-4)
    + 0.04 * Math.log(sL + 1e-4)
    + 0.18 * Math.log(tex + 1e-4)
    + 0.18 * Math.log(sGrid + 1e-4),
  );
}

/** Auto Pairs still uses a light size/shape cue between two live socks. */
export function similarity(a, b) {
  const sAB = bhatt(a.ab, b.ab);
  const sL = bhatt(a.lh, b.lh);
  const sPal = Math.exp(-palDist(a.pal, b.pal) / 14);
  const sMean = Math.exp(-cdist(a.mean[0], a.mean[1], a.mean[2], b.mean[0], b.mean[1], b.mean[2]) / 10);
  const sBand = Math.exp(-bandDist(a.bands, b.bands) / 16);
  const sOri = bhatt(a.ori, b.ori);
  const sEdge = Math.exp(-Math.abs(a.edge - b.edge) / 0.09);
  const sStripe = Math.exp(-Math.abs(a.stripes - b.stripes) / 3);
  const tex = 0.4 * sEdge + 0.35 * sOri + 0.25 * sStripe;
  const areaR = Math.min(a.area, b.area) / Math.max(a.area, b.area, 1);
  const elR = Math.min(a.elong, b.elong) / Math.max(a.elong, b.elong, 1);
  const shape = 0.7 + 0.3 * elR;
  const s = Math.exp(
    0.24 * Math.log(sAB + 1e-4)
    + 0.18 * Math.log(sPal + 1e-4)
    + 0.16 * Math.log(sMean + 1e-4)
    + 0.14 * Math.log(sBand + 1e-4)
    + 0.06 * Math.log(sL + 1e-4)
    + 0.16 * Math.log(tex + 1e-4)
    + 0.06 * Math.log(shape),
  );
  const sizeF = a.held || b.held ? 1 : Math.min(1, 0.55 + areaR);
  return s * sizeF;
}

/** Drop size/shape fields from a stored scan so matching cannot use them. */
export function appearanceOf(d) {
  if (!d) return null;
  return {
    ab: d.ab, lh: d.lh, ori: d.ori, bands: d.bands, grid: d.grid,
    mean: d.mean, pal: d.pal, edge: d.edge, stripes: d.stripes,
  };
}

export function serializeAppearance(d) {
  const a = appearanceOf(d);
  if (!a) return null;
  const arr = (v) => (v && typeof v.length === 'number' ? Array.from(v) : v);
  return {
    ab: arr(a.ab), lh: arr(a.lh), ori: arr(a.ori), bands: arr(a.bands), grid: arr(a.grid),
    mean: arr(a.mean), pal: a.pal, edge: a.edge, stripes: a.stripes,
  };
}

export function parseAppearance(raw) {
  if (!raw) return null;
  const f32 = (v) => (v ? new Float32Array(v) : v);
  return {
    ab: f32(raw.ab), lh: f32(raw.lh), ori: f32(raw.ori), bands: f32(raw.bands), grid: f32(raw.grid),
    mean: raw.mean ? Array.from(raw.mean) : raw.mean,
    pal: raw.pal, edge: raw.edge, stripes: raw.stripes,
  };
}

/** strictness 0..1 -> similarity threshold. */
export function thresholdFor(strictness) {
  return 0.46 + strictness * 0.3; // default 0.55 -> 0.625
}

/**
 * Confident pairs: mutual best matches above threshold that also beat each
 * one's runner-up by a margin. Returns sorted by score (best first).
 */
export function confidentPairs(ids, sim, threshold, margin = 0.02) {
  const n = ids.length;
  const best = new Int32Array(n).fill(-1), bestS = new Float32Array(n), second = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const s = sim[i * n + j];
      if (s > bestS[i]) { second[i] = bestS[i]; bestS[i] = s; best[i] = j; } else if (s > second[i]) second[i] = s;
    }
  }
  const out = [];
  for (let i = 0; i < n; i++) {
    const j = best[i];
    if (j <= i || best[j] !== i) continue;
    const s = bestS[i];
    if (s < threshold) continue;
    const m = Math.min(s - second[i], s - second[j]);
    if (m < margin) continue;
    out.push({ a: ids[i], b: ids[j], score: s, margin: m });
  }
  return out.sort((p, q) => q.score - p.score);
}
