/**
 * Sock descriptor: a mix of robust color, pattern and layout signals.
 *  - ab: soft 8x8 a*b* chroma histogram (after per-frame gray-world WB)
 *  - lh: soft 8-bin lightness histogram, normalized to the frame surface
 *  - pal: 3 dominant Lab colors (tiny k-means) with weights
 *  - bands: 5 mean-Lab bands along the sock's long axis (cuff..toe)
 *  - ori: gradient orientation histogram relative to the long axis
 *  - edge / stripes: texture density and stripe frequency along the axis
 *  - shape: elongation + area
 */
const AB_B = 8, AB_LO = -40, AB_SPAN = 80;
const L_B = 8;
const BANDS = 5;
const PROF = 16;

export function describeRegion(r, planes, w, h, regionMap, refL) {
  const { L, A, B, Ls, As, Bs } = planes;
  const px = r.pixels;
  const ab = new Float32Array(AB_B * AB_B);
  const lh = new Float32Array(L_B);
  const ori = new Float32Array(8);
  const pL = new Float32Array(PROF), pA = new Float32Array(PROF), pB = new Float32Array(PROF), pN = new Float32Array(PROF);
  const ca = Math.cos(r.angle), sa = Math.sin(r.angle);
  const half = Math.max(1, r.major * 1.9);
  // Lightness relative to the surface brightness: survives exposure shifts.
  const lScale = 62 / Math.max(25, refL || 62);
  let edge = 0, interior = 0, mL = 0, mA = 0, mB = 0;
  const step = px.length > 1600 ? 2 : 1;
  const samples = [];
  for (let k = 0; k < px.length; k += step) {
    const i = px[k];
    const x = i % w, y = (i / w) | 0;
    const id = r.id;
    // Skip the 1px rim — it mixes with the surface.
    if (x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1) continue;
    if (regionMap[i - 1] !== id || regionMap[i + 1] !== id || regionMap[i - w] !== id || regionMap[i + w] !== id) continue;
    interior++;
    const l = Math.min(100, Ls[i] * lScale), a = As[i], b = Bs[i];
    mL += l; mA += a; mB += b;
    // Soft chroma histogram.
    const fa = ((a - AB_LO) / AB_SPAN) * AB_B - 0.5, fb = ((b - AB_LO) / AB_SPAN) * AB_B - 0.5;
    const a0 = Math.floor(fa), b0 = Math.floor(fb), ta = fa - a0, tb = fb - b0;
    for (let da = 0; da <= 1; da++) {
      const ai = Math.max(0, Math.min(AB_B - 1, a0 + da));
      const wa = da ? ta : 1 - ta;
      for (let db = 0; db <= 1; db++) {
        const bi = Math.max(0, Math.min(AB_B - 1, b0 + db));
        ab[ai * AB_B + bi] += wa * (db ? tb : 1 - tb);
      }
    }
    const fl = (l / 100) * L_B - 0.5, l0 = Math.floor(fl), tl = fl - l0;
    lh[Math.max(0, Math.min(L_B - 1, l0))] += 1 - tl;
    lh[Math.max(0, Math.min(L_B - 1, l0 + 1))] += tl;
    // Axis profile.
    const u = ((x - r.cx) * ca + (y - r.cy) * sa) / half; // -1..1 roughly
    const pi = Math.max(0, Math.min(PROF - 1, ((u + 1) / 2 * PROF) | 0));
    pL[pi] += l; pA[pi] += a; pB[pi] += b; pN[pi]++;
    // Texture on the unblurred L.
    const gx = L[i + 1] - L[i - 1], gy = L[i + w] - L[i - w];
    const mag = Math.abs(gx) + Math.abs(gy);
    if (mag > 7) {
      edge++;
      let ang = Math.atan2(gy, gx) - r.angle;
      ang = ((ang % Math.PI) + Math.PI) % Math.PI;
      ori[Math.min(7, (ang / Math.PI * 8) | 0)] += mag;
    }
    if ((k & 3) === 0) samples.push(l, a, b);
  }
  const cnt = Math.max(1, interior);
  norm(ab); norm(lh); norm(ori);
  // Bands from the profile (fill empty profile bins from neighbours).
  for (let p = 0; p < PROF; p++) {
    if (pN[p]) { pL[p] /= pN[p]; pA[p] /= pN[p]; pB[p] /= pN[p]; }
  }
  fillGaps(pL, pA, pB, pN, mL / cnt, mA / cnt, mB / cnt);
  const bands = new Float32Array(BANDS * 3);
  for (let bnd = 0; bnd < BANDS; bnd++) {
    const p0 = Math.round(bnd * PROF / BANDS), p1 = Math.round((bnd + 1) * PROF / BANDS);
    let s = 0, l = 0, a = 0, b = 0;
    for (let p = p0; p < p1; p++) { const wgt = pN[p] + 0.5; l += pL[p] * wgt; a += pA[p] * wgt; b += pB[p] * wgt; s += wgt; }
    bands[bnd * 3] = l / s; bands[bnd * 3 + 1] = a / s; bands[bnd * 3 + 2] = b / s;
  }
  // Stripe frequency: sign changes of the detrended L profile with enough amplitude.
  let stripes = 0, last = 0;
  for (let p = 1; p < PROF - 1; p++) {
    const d = pL[p] - (pL[p - 1] + pL[p + 1]) / 2;
    if (Math.abs(d) < 3) continue;
    const s = d > 0 ? 1 : -1;
    if (last && s !== last) stripes++;
    last = s;
  }
  return {
    ab, lh, ori, bands,
    mean: [mL / cnt, mA / cnt, mB / cnt],
    pal: palette(samples, 3),
    edge: edge / cnt,
    stripes,
    elong: r.elongation,
    area: r.area,
    n: interior,
  };
}

function norm(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  if (s > 0) for (let i = 0; i < a.length; i++) a[i] /= s;
}

function fillGaps(pL, pA, pB, pN, dl, da, db) {
  const n = pL.length;
  for (let p = 0; p < n; p++) {
    if (pN[p]) continue;
    let q = -1;
    for (let d = 1; d < n && q < 0; d++) {
      if (p - d >= 0 && pN[p - d]) q = p - d; else if (p + d < n && pN[p + d]) q = p + d;
    }
    if (q >= 0) { pL[p] = pL[q]; pA[p] = pA[q]; pB[p] = pB[q]; } else { pL[p] = dl; pA[p] = da; pB[p] = db; }
  }
}

function palette(s, k) {
  const m = s.length / 3;
  if (!m) return [{ L: 50, a: 0, b: 0, w: 1 }];
  if (m <= k) {
    const out = [];
    for (let i = 0; i < m; i++) out.push({ L: s[i * 3], a: s[i * 3 + 1], b: s[i * 3 + 2], w: 1 / m });
    return out;
  }
  // k-means++-ish deterministic init: darkest, brightest, most chromatic.
  const c = new Float32Array(k * 3);
  let iD = 0, iB = 0, iC = 0, vD = 1e9, vB = -1e9, vC = -1;
  for (let i = 0; i < m; i++) {
    const l = s[i * 3], ch = s[i * 3 + 1] ** 2 + s[i * 3 + 2] ** 2;
    if (l < vD) { vD = l; iD = i; }
    if (l > vB) { vB = l; iB = i; }
    if (ch > vC) { vC = ch; iC = i; }
  }
  [iD, iB, iC].slice(0, k).forEach((idx, j) => { c[j * 3] = s[idx * 3]; c[j * 3 + 1] = s[idx * 3 + 1]; c[j * 3 + 2] = s[idx * 3 + 2]; });
  const acc = new Float32Array(k * 4);
  for (let it = 0; it < 6; it++) {
    acc.fill(0);
    for (let i = 0; i < m; i++) {
      const l = s[i * 3], a = s[i * 3 + 1], b = s[i * 3 + 2];
      let best = 0, bd = 1e12;
      for (let j = 0; j < k; j++) {
        const dl = (l - c[j * 3]) * 0.7, da = a - c[j * 3 + 1], db = b - c[j * 3 + 2];
        const d = dl * dl + da * da + db * db;
        if (d < bd) { bd = d; best = j; }
      }
      acc[best * 4] += l; acc[best * 4 + 1] += a; acc[best * 4 + 2] += b; acc[best * 4 + 3]++;
    }
    for (let j = 0; j < k; j++) {
      const q = acc[j * 4 + 3];
      if (q) { c[j * 3] = acc[j * 4] / q; c[j * 3 + 1] = acc[j * 4 + 1] / q; c[j * 3 + 2] = acc[j * 4 + 2] / q; }
    }
  }
  const out = [];
  for (let j = 0; j < k; j++) {
    const q = acc[j * 4 + 3];
    if (q) out.push({ L: c[j * 3], a: c[j * 3 + 1], b: c[j * 3 + 2], w: q / m });
  }
  return out.sort((x, y) => y.w - x.w);
}
