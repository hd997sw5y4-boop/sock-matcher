import { rgbaToLab } from './color.js';
import {
  estimateBackground, buildDelta, otsuThreshold, thresholdMask,
  cleanupMask, findBlobs, blobContour,
} from './detect.js';
import { describeBlob } from './describe.js';
import { pairSocks } from './match.js';

export const WORK_WIDTH = 400;

export function createProcessor() {
  let work = document.createElement('canvas');
  const ctx = work.getContext('2d', { willReadFrequently: true });
  let lab = new Float32Array(0);
  let delta = new Float32Array(0);
  let mask = new Uint8Array(0);
  let scratch = new Uint8Array(0);
  let lockedBg = null;

  function resize(w, h) {
    if (work.width !== w || work.height !== h) {
      work.width = w;
      work.height = h;
      const n = w * h;
      lab = new Float32Array(n * 3);
      delta = new Float32Array(n);
      mask = new Uint8Array(n);
      scratch = new Uint8Array(n);
    }
  }

  function lockSurface(bg) { lockedBg = bg; }
  function unlockSurface() { lockedBg = null; }
  function isLocked() { return !!lockedBg; }

  /**
   * @param {CanvasImageSource} source
   * @param {{sensitivity:number, strictness:number, srcW:number, srcH:number}} opts
   *        sensitivity/strictness in 0..1
   */
  function run(source, opts) {
    const srcW = opts.srcW;
    const srcH = opts.srcH;
    const w = WORK_WIDTH;
    const h = Math.max(8, Math.round(WORK_WIDTH * (srcH / srcW)));
    resize(w, h);
    ctx.drawImage(source, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h);
    rgbaToLab(img.data, w * h, lab);

    const bg = lockedBg || estimateBackground(lab, w, h);
    buildDelta(lab, w, h, bg, delta);

    // Sensitivity: 0 = picky (higher threshold), 1 = greedy (lower threshold).
    const auto = otsuThreshold(delta, w * h);
    const offset = (0.5 - opts.sensitivity) * 18;
    const t = Math.max(6, auto + offset);
    thresholdMask(delta, w * h, t, mask);
    cleanupMask(mask, w, h, scratch);

    const blobs = findBlobs(mask, w, h);
    for (const b of blobs) b.contour = blobContour(b, mask, w, h);
    const descs = blobs.map((b) => describeBlob(b, img.data, lab, mask, w, h));
    const pairing = pairSocks(descs, opts.strictness);

    return { w, h, srcW, srcH, bg, threshold: t, blobs, descs, pairing, mask };
  }

  /** Sample a locked surface color from a work-space point. */
  function sampleSurface(wx, wy) {
    const w = work.width, h = work.height;
    if (!w) return null;
    const r = 6;
    const Ls = [], As = [], Bs = [];
    for (let y = Math.max(0, wy - r); y <= Math.min(h - 1, wy + r); y++) {
      for (let x = Math.max(0, wx - r); x <= Math.min(w - 1, wx + r); x++) {
        const i = (y * w + x) * 3;
        Ls.push(lab[i]); As.push(lab[i + 1]); Bs.push(lab[i + 2]);
      }
    }
    if (!Ls.length) return null;
    const mid = (arr) => {
      const a = arr.slice().sort((x, y) => x - y);
      return a[a.length >> 1];
    };
    const bg = { L: mid(Ls), a: mid(As), b: mid(Bs) };
    lockedBg = bg;
    return bg;
  }

  return { run, lockSurface, unlockSurface, isLocked, sampleSurface, canvas: work };
}
