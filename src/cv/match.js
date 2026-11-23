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
  // Weighted geometric mean so one strongly disagreeing cue vetoes a match.
  const s = Math.exp(
    0.24 * Math.log(sAB + 1e-4)
    + 0.18 * Math.log(sPal + 1e-4)
    + 0.16 * Math.log(sMean + 1e-4)
    + 0.14 * Math.log(sBand + 1e-4)
    + 0.06 * Math.log(sL + 1e-4)
    + 0.16 * Math.log(tex + 1e-4)
    + 0.06 * Math.log(shape),
  );
  // Mates show a similar visible size; a held-up scan is at another scale.
  const sizeF = a.held || b.held ? 1 : Math.min(1, 0.55 + areaR);
  return s * sizeF;
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
