/**
 * Fast sRGB -> CIELAB via a 6-bit-per-channel lookup table, with a per-frame
 * gray-world white-balance gain baked into small per-channel index tables.
 * Pure JS (no DOM) so it runs in a Worker, on the main thread, or in Node.
 */
let LUT = null; // Float32Array(64^3 * 3)

function labF(t) {
  return t > 0.008856 ? Math.cbrt(t) : 7.787037 * t + 16 / 116;
}

export function rgbToLab(r, g, b) {
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const R = lin(r), G = lin(g), B = lin(b);
  const x = labF((R * 0.4124564 + G * 0.3575761 + B * 0.1804375) / 0.95047);
  const y = labF(R * 0.2126729 + G * 0.7151522 + B * 0.0721750);
  const z = labF((R * 0.0193339 + G * 0.1191920 + B * 0.9503041) / 1.08883);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}

function buildLut() {
  LUT = new Float32Array(64 * 64 * 64 * 3);
  let k = 0;
  for (let r = 0; r < 64; r++) {
    for (let g = 0; g < 64; g++) {
      for (let b = 0; b < 64; b++) {
        const lab = rgbToLab(r * 4 + 2, g * 4 + 2, b * 4 + 2);
        LUT[k++] = lab[0]; LUT[k++] = lab[1]; LUT[k++] = lab[2];
      }
    }
  }
}

const idxR = new Int32Array(256);
const idxG = new Int32Array(256);
const idxB = new Int32Array(256);

/**
 * Convert RGBA -> planar L, A, B with gray-world correction.
 * `strength` 0..1 controls how far toward neutral the frame is pulled.
 * Returns the gains used.
 */
export function rgbaToLabPlanes(rgba, n, L, A, B, strength = 0.75) {
  if (!LUT) buildLut();
  let sr = 0, sg = 0, sb = 0, cnt = 0;
  const step = n > 20000 ? 7 : 1;
  for (let i = 0; i < n; i += step) {
    const p = i * 4;
    sr += rgba[p]; sg += rgba[p + 1]; sb += rgba[p + 2];
    cnt++;
  }
  sr /= cnt; sg /= cnt; sb /= cnt;
  const gray = (sr + sg + sb) / 3 || 1;
  const gain = (m) => Math.max(0.75, Math.min(1.35, (gray / Math.max(1, m)) ** strength));
  const gr = gain(sr), gg = gain(sg), gb = gain(sb);
  for (let v = 0; v < 256; v++) {
    idxR[v] = Math.min(63, (v * gr) >> 2) * 4096 * 3;
    idxG[v] = Math.min(63, (v * gg) >> 2) * 64 * 3;
    idxB[v] = Math.min(63, (v * gb) >> 2) * 3;
  }
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const k = idxR[rgba[p]] + idxG[rgba[p + 1]] + idxB[rgba[p + 2]];
    L[i] = LUT[k]; A[i] = LUT[k + 1]; B[i] = LUT[k + 2];
  }
  return [gr, gg, gb];
}

/** Separable 3x3 box blur of a float plane into dst (tmp is scratch). */
export function blur3(src, dst, tmp, w, h) {
  for (let y = 0; y < h; y++) {
    const o = y * w;
    tmp[o] = (src[o] * 2 + src[o + 1]) / 3;
    for (let x = 1; x < w - 1; x++) tmp[o + x] = (src[o + x - 1] + src[o + x] + src[o + x + 1]) / 3;
    tmp[o + w - 1] = (src[o + w - 1] * 2 + src[o + w - 2]) / 3;
  }
  for (let x = 0; x < w; x++) {
    dst[x] = (tmp[x] * 2 + tmp[x + w]) / 3;
    for (let y = 1; y < h - 1; y++) {
      const i = y * w + x;
      dst[i] = (tmp[i - w] + tmp[i] + tmp[i + w]) / 3;
    }
    const last = (h - 1) * w + x;
    dst[last] = (tmp[last] * 2 + tmp[last - w]) / 3;
  }
}
