/** sRGB → linear lookup. */
const LIN = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  LIN[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function labF(t) {
  return t > 0.008856 ? Math.cbrt(t) : 7.787037 * t + 16 / 116;
}

/** Convert a 0–255 sRGB triple to CIELAB (D65). */
export function rgbToLab(r, g, b) {
  const R = LIN[r], G = LIN[g], B = LIN[b];
  const x = labF((R * 0.4124564 + G * 0.3575761 + B * 0.1804375) / 0.95047);
  const y = labF(R * 0.2126729 + G * 0.7151522 + B * 0.0721750);
  const z = labF((R * 0.0193339 + G * 0.1191920 + B * 0.9503041) / 1.08883);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}

export function rgbaToLab(rgba, count, lab) {
  let pi = 0, li = 0;
  for (let i = 0; i < count; i++) {
    const r = rgba[pi], g = rgba[pi + 1], b = rgba[pi + 2];
    pi += 4;
    const R = LIN[r], G = LIN[g], B = LIN[b];
    const x = labF((R * 0.4124564 + G * 0.3575761 + B * 0.1804375) / 0.95047);
    const y = labF(R * 0.2126729 + G * 0.7151522 + B * 0.0721750);
    const z = labF((R * 0.0193339 + G * 0.1191920 + B * 0.9503041) / 1.08883);
    lab[li++] = 116 * y - 16;
    lab[li++] = 500 * (x - y);
    lab[li++] = 200 * (y - z);
  }
  return lab;
}

/** Hue in [0,1), saturation, value. */
export function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d + 6) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h /= 6;
  }
  return [h, max === 0 ? 0 : d / max, max];
}

export function histogramIntersection(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] < b[i] ? a[i] : b[i];
  return s;
}

export function l1Normalize(hist) {
  let s = 0;
  for (let i = 0; i < hist.length; i++) s += hist[i];
  if (s <= 0) return hist;
  const inv = 1 / s;
  for (let i = 0; i < hist.length; i++) hist[i] *= inv;
  return hist;
}

/** Weighted Lab distance that downplays shadows (lightness). */
export function chromaHeavyDist(dL, da, db) {
  return Math.sqrt(0.35 * dL * dL + da * da + db * db);
}

export function labDist(L1, a1, b1, L2, a2, b2) {
  const dL = L1 - L2, da = a1 - a2, db = b1 - b2;
  return Math.sqrt(dL * dL + da * da + db * db);
}

/** Median of a copied numeric array. */
export function median(values) {
  if (!values.length) return 0;
  const a = values.slice().sort((x, y) => x - y);
  const mid = a.length >> 1;
  return a.length % 2 ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
}
