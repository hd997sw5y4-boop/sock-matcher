import { rgbToHsv, rgbToLab, l1Normalize, labDist } from './color.js';
import { erodePixels } from './detect.js';

const H_BINS = 16;
const S_BINS = 8;
const L_BINS = 8;
const AB_BINS = 6; // 6x6 a*b* grid

export function describeBlob(blob, rgba, lab, mask, w, h) {
  const inner = erodePixels(blob.pixels, mask, w, h, blob.area > 400 ? 2 : 1);
  const histH = new Float32Array(H_BINS);
  const histS = new Float32Array(S_BINS);
  const histL = new Float32Array(L_BINS);
  const histAB = new Float32Array(AB_BINS * AB_BINS);
  const samples = [];

  let sumL = 0, sumA = 0, sumB = 0;
  let edge = 0, gSum = 0, gSum2 = 0, gN = 0;
  const orient = new Float32Array(8);

  for (const i of inner) {
    const p = i * 4;
    const r = rgba[p], g = rgba[p + 1], b = rgba[p + 2];
    const [hh, ss, vv] = rgbToHsv(r, g, b);
    const li = i * 3;
    const L = lab[li], A = lab[li + 1], B = lab[li + 2];
    sumL += L; sumA += A; sumB += B;

    // Hue only counts when the pixel actually has color.
    const hueW = ss * Math.min(1, vv / 0.15);
    histH[Math.min(H_BINS - 1, (hh * H_BINS) | 0)] += hueW;
    histS[Math.min(S_BINS - 1, (ss * S_BINS) | 0)] += 1;
    histL[Math.min(L_BINS - 1, (Math.max(0, Math.min(99, L)) / 100 * L_BINS) | 0)] += 1;

    const ai = Math.max(0, Math.min(AB_BINS - 1, (((A + 80) / 160) * AB_BINS) | 0));
    const bi = Math.max(0, Math.min(AB_BINS - 1, (((B + 80) / 160) * AB_BINS) | 0));
    histAB[ai * AB_BINS + bi] += 1;

    if (ss > 0.12 && vv > 0.12) samples.push([L, A, B]);

    const x = i % w, y = (i / w) | 0;
    if (x > blob.minX && x < blob.maxX && y > blob.minY && y < blob.maxY) {
      const lR = lab[((y * w + (x + 1)) * 3)];
      const lL = lab[((y * w + (x - 1)) * 3)];
      const lD = lab[(((y + 1) * w + x) * 3)];
      const lU = lab[(((y - 1) * w + x) * 3)];
      const gx = lR - lL;
      const gy = lD - lU;
      const mag = Math.hypot(gx, gy);
      gSum += mag;
      gSum2 += mag * mag;
      gN++;
      if (mag > 6) {
        edge++;
        let ang = Math.atan2(gy, gx);
        if (ang < 0) ang += Math.PI; // unsigned orientation
        orient[Math.min(7, (ang / Math.PI * 8) | 0)] += mag;
      }
    }
  }

  const n = inner.length || 1;
  l1Normalize(histH);
  l1Normalize(histS);
  l1Normalize(histL);
  l1Normalize(histAB);
  l1Normalize(orient);

  const mean = [sumL / n, sumA / n, sumB / n];
  const palette = kMeansPalette(samples.length ? samples : [[mean[0], mean[1], mean[2]]], 3);
  const edgeDensity = edge / n;
  const gradMean = gN ? gSum / gN : 0;
  const gradVar = gN ? Math.max(0, gSum2 / gN - gradMean * gradMean) : 0;

  return {
    histH, histS, histL, histAB, mean, palette,
    edgeDensity, gradMean, gradVar, orient,
    aspect: blob.aspect,
    area: blob.area,
    elongation: blob.elongation,
  };
}

function kMeansPalette(samples, k) {
  if (samples.length <= k) {
    return samples.map((s) => ({ L: s[0], a: s[1], b: s[2], w: 1 / Math.max(1, samples.length) }));
  }
  // Init: spread along first 3 unique-ish samples.
  const centers = [];
  const step = Math.max(1, (samples.length / k) | 0);
  for (let i = 0; i < k; i++) {
    const s = samples[Math.min(samples.length - 1, i * step)];
    centers.push([s[0], s[1], s[2]]);
  }
  const assign = new Int16Array(samples.length);
  const counts = new Float32Array(k);
  for (let iter = 0; iter < 7; iter++) {
    counts.fill(0);
    const acc = Array.from({ length: k }, () => [0, 0, 0]);
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      let best = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const d = labDist(s[0], s[1], s[2], centers[c][0], centers[c][1], centers[c][2]);
        if (d < bestD) { bestD = d; best = c; }
      }
      assign[i] = best;
      counts[best]++;
      acc[best][0] += s[0]; acc[best][1] += s[1]; acc[best][2] += s[2];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        centers[c][0] = acc[c][0] / counts[c];
        centers[c][1] = acc[c][1] / counts[c];
        centers[c][2] = acc[c][2] / counts[c];
      }
    }
  }
  const total = samples.length;
  const palette = [];
  for (let c = 0; c < k; c++) {
    palette.push({ L: centers[c][0], a: centers[c][1], b: centers[c][2], w: counts[c] / total });
  }
  palette.sort((x, y) => y.w - x.w);
  return palette;
}

export function paletteDistance(p, q) {
  // Earth-mover-ish: each color in p to nearest in q, weighted, symmetric.
  const oneWay = (a, b) => {
    let s = 0;
    for (const ca of a) {
      let best = Infinity;
      for (const cb of b) {
        const d = labDist(ca.L, ca.a, ca.b, cb.L, cb.a, cb.b);
        if (d < best) best = d;
      }
      s += ca.w * best;
    }
    return s;
  };
  return 0.5 * (oneWay(p, q) + oneWay(q, p));
}
