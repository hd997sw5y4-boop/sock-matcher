import { chromaHeavyDist, median } from './color.js';

const tmpHist = new Float32Array(64);

export function estimateBackground(lab, w, h, borderFrac = 0.1) {
  const bw = Math.max(2, Math.round(w * borderFrac));
  const bh = Math.max(2, Math.round(h * borderFrac));
  const Ls = [], As = [], Bs = [];
  const push = (x, y) => {
    const i = (y * w + x) * 3;
    Ls.push(lab[i]); As.push(lab[i + 1]); Bs.push(lab[i + 2]);
  };
  for (let y = 0; y < h; y++) {
    const onV = y < bh || y >= h - bh;
    for (let x = 0; x < w; x++) {
      if (onV || x < bw || x >= w - bw) push(x, y);
    }
  }
  return { L: median(Ls), a: median(As), b: median(Bs) };
}

export function buildDelta(lab, w, h, bg, out) {
  const n = w * h;
  let max = 0;
  for (let i = 0, p = 0; i < n; i++, p += 3) {
    const d = chromaHeavyDist(lab[p] - bg.L, lab[p + 1] - bg.a, lab[p + 2] - bg.b);
    out[i] = d;
    if (d > max) max = d;
  }
  return max;
}

/** Otsu threshold on a float map, with optional clip of the upper tail. */
export function otsuThreshold(map, n, bins = 64) {
  tmpHist.fill(0);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = map[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!(max > min)) return min + 1;
  const scale = (bins - 1) / (max - min);
  for (let i = 0; i < n; i++) tmpHist[((map[i] - min) * scale) | 0]++;
  let sum = 0;
  for (let i = 0; i < bins; i++) sum += i * tmpHist[i];
  let sumB = 0, wB = 0, best = 0, t = 0;
  const total = n;
  for (let i = 0; i < bins; i++) {
    wB += tmpHist[i];
    if (!wB) continue;
    const wF = total - wB;
    if (!wF) break;
    sumB += i * tmpHist[i];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) {
      best = between;
      t = i;
    }
  }
  return min + (t + 0.5) / scale;
}

export function thresholdMask(map, n, t, mask) {
  for (let i = 0; i < n; i++) mask[i] = map[i] >= t ? 1 : 0;
}

function morph(src, w, h, dilate, dst, r = 1) {
  dst.fill(0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let val = dilate ? 0 : 1;
      outer: for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          if (dx * dx + dy * dy > r * r + 0.5) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const v = src[yy * w + xx];
          if (dilate) {
            if (v) { val = 1; break outer; }
          } else if (!v) { val = 0; break outer; }
        }
      }
      dst[y * w + x] = val;
    }
  }
}

/** Close then open to fill holes and drop speckles. */
export function cleanupMask(mask, w, h, scratch) {
  // Close with a slightly larger radius so striped socks stay one blob,
  // then open to drop speckles.
  morph(mask, w, h, true, scratch, 3);
  morph(scratch, w, h, false, mask, 3);
  morph(mask, w, h, false, scratch, 1);
  morph(scratch, w, h, true, mask, 1);
}

const N4 = [[1, 0], [-1, 0], [0, 1], [0, -1]];

export function findBlobs(mask, w, h, opts = {}) {
  const minArea = opts.minArea ?? Math.max(60, (w * h) * 0.0022);
  const maxArea = opts.maxArea ?? (w * h) * 0.45;
  const seen = new Uint8Array(w * h);
  const blobs = [];
  const stack = [];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const start = y * w + x;
      if (!mask[start] || seen[start]) continue;
      stack.length = 0;
      stack.push(start);
      seen[start] = 1;

      let area = 0;
      let sumX = 0, sumY = 0;
      let minX = x, maxX = x, minY = y, maxY = y;
      let sumXX = 0, sumYY = 0, sumXY = 0;
      const pixels = [];

      while (stack.length) {
        const i = stack.pop();
        const px = i % w;
        const py = (i / w) | 0;
        area++;
        sumX += px;
        sumY += py;
        sumXX += px * px;
        sumYY += py * py;
        sumXY += px * py;
        pixels.push(i);
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
        for (const [dx, dy] of N4) {
          const nx = px + dx, ny = py + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const ni = ny * w + nx;
          if (!mask[ni] || seen[ni]) continue;
          seen[ni] = 1;
          stack.push(ni);
        }
      }

      if (area < minArea || area > maxArea) continue;

      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      const aspect = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
      if (aspect > 5.5) continue;
      const solidity = area / (bw * bh);
      if (solidity < 0.22) continue;

      const cx = sumX / area;
      const cy = sumY / area;
      const covXX = sumXX / area - cx * cx;
      const covYY = sumYY / area - cy * cy;
      const covXY = sumXY / area - cx * cy;
      const trace = covXX + covYY;
      const det = covXX * covYY - covXY * covXY;
      const disc = Math.sqrt(Math.max(0, trace * trace - 4 * det));
      const ev1 = 0.5 * (trace + disc);
      const ev2 = 0.5 * (trace - disc);
      const elongation = Math.sqrt(Math.max(ev1, 1e-6)) / Math.sqrt(Math.max(ev2, 1e-6));
      let angle = 0.5 * Math.atan2(2 * covXY, covXX - covYY);

      blobs.push({
        area, cx, cy, minX, minY, maxX, maxY, bw, bh,
        aspect, solidity, elongation, angle, pixels,
      });
    }
  }

  blobs.sort((a, b) => b.area - a.area);
  return blobs;
}

/** Boundary walk → simplified polygon in work-pixel space. */
export function blobContour(blob, mask, w, h, maxPts = 36) {
  const { minX, minY, maxX, maxY } = blob;
  let sx = -1, sy = -1;
  outer: for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (mask[y * w + x]) { sx = x; sy = y; break outer; }
    }
  }
  if (sx < 0) {
    return [
      [blob.minX, blob.minY], [blob.maxX, blob.minY],
      [blob.maxX, blob.maxY], [blob.minX, blob.maxY],
    ];
  }

  // Clockwise 8-neighborhood, start west of first pixel.
  const dx = [1, 1, 0, -1, -1, -1, 0, 1];
  const dy = [0, 1, 1, 1, 0, -1, -1, -1];
  const pts = [];
  let x = sx, y = sy, dir = 4;
  const limit = blob.area * 8 + 16;
  for (let n = 0; n < limit; n++) {
    pts.push([x, y]);
    let found = false;
    const start = (dir + 6) % 8; // turn left first
    for (let k = 0; k < 8; k++) {
      const d = (start + k) % 8;
      const nx = x + dx[d], ny = y + dy[d];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      if (mask[ny * w + nx]) {
        x = nx; y = ny; dir = d; found = true; break;
      }
    }
    if (!found) break;
    if (x === sx && y === sy && pts.length > 8) break;
  }

  if (pts.length <= maxPts) return pts;
  const step = pts.length / maxPts;
  const out = [];
  for (let i = 0; i < maxPts; i++) out.push(pts[(i * step) | 0]);
  return out;
}

export function erodePixels(pixels, mask, w, h, radius = 1) {
  const set = new Set(pixels);
  const out = [];
  for (const i of pixels) {
    const x = i % w, y = (i / w) | 0;
    let ok = true;
    for (let dy = -radius; dy <= radius && ok; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy > radius * radius + 0.5) continue;
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h || !mask[yy * w + xx]) {
          ok = false;
          break;
        }
      }
    }
    if (ok) out.push(i);
  }
  return out.length ? out : pixels;
}
