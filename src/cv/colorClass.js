/**
 * Lightweight color buckets for Color Sort. Works on the same low-res Lab
 * buffer as the classic detector (about 320 px), so it stays inside the
 * battery-saver budget. No model, no full-frame embeddings.
 */

export const COLOR_CHIPS = [
  { id: 'pink', label: 'Pink', swatch: '#f4a4c0', tint: '#ff7ab2' },
  { id: 'white', label: 'White', swatch: '#f3f1ec', tint: '#f5f5f7' },
  { id: 'grey', label: 'Grey', swatch: '#8e8e93', tint: '#c7c7cc' },
  { id: 'black', label: 'Black', swatch: '#2c2c2e', tint: '#5a5a5e' },
  { id: 'blue', label: 'Blue', swatch: '#3b82f6', tint: '#0a84ff' },
  { id: 'green', label: 'Green', swatch: '#34d399', tint: '#30d158' },
  { id: 'purple', label: 'Purple', swatch: '#a78bfa', tint: '#bf5af2' },
  { id: 'red', label: 'Red', swatch: '#ef4444', tint: '#ff453a' },
  { id: 'yellow', label: 'Yellow', swatch: '#f5d76e', tint: '#ffd60a' },
  { id: 'multi', label: 'Multi', swatch: '#fff', tint: '#ff9f0a' },
  { id: 'pick', label: 'Tap to pick', swatch: '#fff', tint: '#64d2ff' },
];

export const CHIP_BY_ID = Object.fromEntries(COLOR_CHIPS.map((c) => [c.id, c]));

/** Named Lab prototypes for nearest-bucket classification. */
const PROTOS = [
  { id: 'white', L: 92, a: 0, b: 4 },
  { id: 'white', L: 84, a: 3, b: 8 },
  { id: 'grey', L: 62, a: 0, b: 1 },
  { id: 'grey', L: 45, a: 0, b: 0 },
  { id: 'black', L: 18, a: 0, b: 0 },
  { id: 'black', L: 10, a: 0, b: 0 },
  { id: 'pink', L: 78, a: 28, b: 6 },
  { id: 'pink', L: 70, a: 38, b: 8 },
  { id: 'pink', L: 82, a: 18, b: 4 },
  { id: 'red', L: 48, a: 52, b: 28 },
  { id: 'red', L: 40, a: 48, b: 22 },
  { id: 'red', L: 55, a: 45, b: 38 }, // coral / orange-red
  { id: 'yellow', L: 86, a: 2, b: 72 },
  { id: 'yellow', L: 78, a: 12, b: 58 },
  { id: 'yellow', L: 70, a: 28, b: 52 }, // mustard / gold
  { id: 'green', L: 58, a: -40, b: 28 },
  { id: 'green', L: 48, a: -32, b: 18 },
  { id: 'green', L: 76, a: -28, b: 8 }, // mint
  { id: 'green', L: 50, a: -32, b: -6 }, // teal
  { id: 'blue', L: 42, a: 8, b: -42 },
  { id: 'blue', L: 58, a: -6, b: -32 },
  { id: 'blue', L: 72, a: -8, b: -22 }, // sky
  { id: 'blue', L: 24, a: 10, b: -30 }, // navy
  { id: 'purple', L: 48, a: 38, b: -32 },
  { id: 'purple', L: 62, a: 28, b: -22 },
  { id: 'purple', L: 38, a: 32, b: -18 },
];

const CHROMATIC = new Set(['pink', 'red', 'yellow', 'green', 'blue', 'purple']);

export function chromaOf(a, b) {
  return Math.hypot(a, b);
}

export function nameFromLab(L, a, b) {
  const c = chromaOf(a, b);
  if (c < 12) {
    if (L >= 74) return 'white';
    if (L <= 28) return 'black';
    return 'grey';
  }
  if (L >= 80 && c < 18) return 'white';
  if (L <= 22 && c < 20) return 'black';
  // Light desaturated pinks (kids' socks) that would otherwise fall in grey.
  if (L >= 62 && a > 8 && Math.abs(b) < 16 && c < 22) return 'pink';

  let best = 'grey', bestD = 1e9;
  for (const p of PROTOS) {
    const dL = (L - p.L) * 0.65;
    const d = dL * dL + (a - p.a) ** 2 + (b - p.b) ** 2;
    if (d < bestD) { bestD = d; best = p.id; }
  }
  return best;
}

export function deltaE(L1, a1, b1, L2, a2, b2) {
  const dL = (L1 - L2) * 0.7;
  return Math.sqrt(dL * dL + (a1 - a2) ** 2 + (b1 - b2) ** 2);
}

function inRoi(x, y, roi, h) {
  if (!roi) return y >= 0 && y < h;
  return y >= (roi.y0 ?? 0) && y <= (roi.y1 ?? h);
}

/**
 * Sample a small window around a tap and return a picked color.
 */
export function pickColorAt(x, y, planes, seg, w, h) {
  const { Ls, As, Bs } = planes;
  const cx = Math.max(0, Math.min(w - 1, Math.round(x)));
  const cy = Math.max(0, Math.min(h - 1, Math.round(y)));
  const fg = seg.fg;
  const rad = 4;
  const samples = [];
  for (let yy = cy - rad; yy <= cy + rad; yy++) {
    if (yy < 0 || yy >= h) continue;
    for (let xx = cx - rad; xx <= cx + rad; xx++) {
      if (xx < 0 || xx >= w) continue;
      const i = yy * w + xx;
      if (fg && !fg[i]) continue;
      samples.push(Ls[i], As[i], Bs[i]);
    }
  }
  // If the tap missed the foreground mask, take raw neighbours.
  if (samples.length < 9) {
    samples.length = 0;
    for (let yy = cy - 2; yy <= cy + 2; yy++) {
      if (yy < 0 || yy >= h) continue;
      for (let xx = cx - 2; xx <= cx + 2; xx++) {
        if (xx < 0 || xx >= w) continue;
        const i = yy * w + xx;
        samples.push(Ls[i], As[i], Bs[i]);
      }
    }
  }
  if (!samples.length) return null;
  const n = samples.length / 3;
  // Median L,a,b (sort each channel).
  const LsA = [], AsA = [], BsA = [];
  for (let k = 0; k < n; k++) {
    LsA.push(samples[k * 3]); AsA.push(samples[k * 3 + 1]); BsA.push(samples[k * 3 + 2]);
  }
  LsA.sort((p, q) => p - q); AsA.sort((p, q) => p - q); BsA.sort((p, q) => p - q);
  const mid = n >> 1;
  const L = LsA[mid], a = AsA[mid], b = BsA[mid];
  return { L, a, b, name: nameFromLab(L, a, b), n };
}

function starOutline(px, w, cx, cy, bins = 24) {
  const rad = new Float32Array(bins);
  for (let k = 0; k < px.length; k++) {
    const i = px[k];
    const dx = (i % w) - cx, dy = ((i / w) | 0) - cy;
    const d2 = dx * dx + dy * dy;
    const b = (((Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI) * bins) | 0) % bins;
    if (d2 > rad[b]) rad[b] = d2;
  }
  const sm = new Float32Array(bins);
  for (let b = 0; b < bins; b++) sm[b] = Math.sqrt(rad[b]);
  for (let b = 0; b < bins; b++) {
    if (sm[b] > 0) continue;
    let l = 1; while (l < bins && sm[(b - l + bins) % bins] === 0) l++;
    let r = 1; while (r < bins && sm[(b + r) % bins] === 0) r++;
    sm[b] = (sm[(b - l + bins) % bins] + sm[(b + r) % bins]) / 2;
  }
  const out = new Float32Array(bins * 2);
  for (let b = 0; b < bins; b++) {
    const v = (sm[(b - 1 + bins) % bins] + 2 * sm[b] + sm[(b + 1) % bins]) / 4 + 1.1;
    const ang = (b + 0.5) / bins * 2 * Math.PI - Math.PI;
    out[b * 2] = Math.cos(ang) * v;
    out[b * 2 + 1] = Math.sin(ang) * v;
  }
  return out;
}

function componentsOf(mask, w, h, roi, minArea, maxArea) {
  const n = w * h;
  const seen = new Uint8Array(n);
  const hits = [];
  const qx = new Int32Array(n);
  for (let start = 0; start < n; start++) {
    if (!mask[start] || seen[start]) continue;
    const sy0 = (start / w) | 0;
    if (!inRoi(start % w, sy0, roi, h)) continue;
    let head = 0, tail = 0;
    qx[tail++] = start;
    seen[start] = 1;
    const px = [];
    let minX = w, minY = h, maxX = 0, maxY = 0, sx = 0, sy = 0;
    while (head < tail) {
      const i = qx[head++];
      const x = i % w, y = (i / w) | 0;
      if (!inRoi(x, y, roi, h)) continue;
      px.push(i);
      sx += x; sy += y;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
      for (const j of nb) {
        if (j < 0 || seen[j] || !mask[j]) continue;
        seen[j] = 1;
        qx[tail++] = j;
      }
    }
    const area = px.length;
    if (area < minArea || area > maxArea) continue;
    const bw = maxX - minX + 1, bh = maxY - minY + 1;
    const solidity = area / (bw * bh);
    if (solidity < 0.20) continue;
    const elong = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
    if (elong > 6.5) continue;
    if (bw < 4 || bh < 4) continue;
    // Huge blobs that hug the frame edge are usually floor or rug.
    const edge = minX <= 0 || minY <= 0 || maxX >= w - 1 || maxY >= h - 1;
    if (edge && area > maxArea * 0.35) continue;
    const cx = sx / area, cy = sy / area;
    hits.push({
      cx, cy, area,
      box: [minX, minY, maxX, maxY],
      outline: starOutline(px, w, cx, cy, area > 80 ? 28 : 18),
    });
  }
  return hits;
}

function regionIsMulti(r, planes, w) {
  if (!r?.pixels || r.pixels.length < 12) return false;
  const { Ls, As, Bs } = planes;
  const counts = new Map();
  const step = r.pixels.length > 400 ? 2 : 1;
  let n = 0, chromatic = 0;
  for (let k = 0; k < r.pixels.length; k += step) {
    const i = r.pixels[k];
    const name = nameFromLab(Ls[i], As[i], Bs[i]);
    counts.set(name, (counts.get(name) || 0) + 1);
    n++;
    if (CHROMATIC.has(name)) chromatic++;
  }
  const significant = [...counts.entries()].filter(([, c]) => c / n >= 0.16);
  const chromaNames = significant.filter(([id]) => CHROMATIC.has(id));
  // Two different hues, or a chromatic color plus a clearly different neutral.
  if (chromaNames.length >= 2) return true;
  if (chromaNames.length >= 1 && significant.length >= 2 && chromatic / n >= 0.2) return true;
  return false;
}

/**
 * Find every sock-like blob matching `query`.
 * query = { id } or { id: 'pick', L, a, b }
 */
export function collectColorHits(planes, seg, query, { roi } = {}) {
  const w = seg.w, h = seg.h;
  const n = w * h;
  const minArea = Math.max(16, Math.round(n * 0.00035));
  const maxArea = Math.round(n * 0.12);
  const fg = seg.fg;
  const { Ls, As, Bs } = planes;
  const mask = new Uint8Array(n);
  const id = query?.id || 'pink';

  if (id === 'multi') {
    const hits = [];
    for (const r of seg.regions || []) {
      if (!inRoi(r.cx, r.cy, roi, h)) continue;
      if (r.area < minArea || r.area > maxArea) continue;
      if (!regionIsMulti(r, planes, w)) continue;
      hits.push({
        id: r.id,
        cx: r.cx,
        cy: r.cy,
        area: r.area,
        box: [r.minX, r.minY, r.maxX, r.maxY],
        outline: r.outline ? (Array.isArray(r.outline) ? r.outline : Array.from(r.outline)) : starOutline(r.pixels, w, r.cx, r.cy),
      });
    }
    return hits;
  }

  if (id === 'pick' && query && Number.isFinite(query.L)) {
    const thr = 16;
    for (let i = 0; i < n; i++) {
      if (fg && !fg[i]) continue;
      if (deltaE(Ls[i], As[i], Bs[i], query.L, query.a, query.b) <= thr) mask[i] = 1;
    }
  } else {
    for (let i = 0; i < n; i++) {
      if (fg && !fg[i]) continue;
      if (nameFromLab(Ls[i], As[i], Bs[i]) === id) mask[i] = 1;
    }
  }

  return componentsOf(mask, w, h, roi, minArea, maxArea);
}

export function chipTint(id) {
  return CHIP_BY_ID[id]?.tint || '#64d2ff';
}
