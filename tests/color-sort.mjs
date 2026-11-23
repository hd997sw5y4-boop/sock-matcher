/**
 * Color Sort unit checks: named Lab buckets, and a private v1420 standing-height
 * pile (never published) should light up many socks of a chosen color, with no 1–3 cap.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createEngine } from '../src/cv/engine.js';
import { nameFromLab, collectColorHits, deltaE, COLOR_CHIPS } from '../src/cv/colorClass.js';
import { rgbToLab } from '../src/cv/lab.js';

let failed = 0;
function assert(cond, msg) {
  if (cond) console.log('ok:', msg);
  else { failed++; console.log('FAIL:', msg); }
}

{
  const cases = [
    ['white', 245, 245, 242],
    ['grey', 140, 140, 142],
    ['black', 20, 20, 22],
    ['pink', 240, 160, 190],
    ['red', 180, 30, 35],
    ['yellow', 230, 200, 40],
    ['green', 30, 150, 70],
    ['blue', 40, 90, 200],
    ['purple', 120, 60, 180],
  ];
  for (const [name, r, g, b] of cases) {
    const lab = rgbToLab(r, g, b);
    const got = nameFromLab(lab[0], lab[1], lab[2]);
    assert(got === name, `${name} rgb(${r},${g},${b}) Lab=${lab.map((x)=>x.toFixed(1)).join(',')} -> ${got}`);
  }
}

{
  assert(COLOR_CHIPS.some((c) => c.id === 'pick'), 'includes tap-to-pick');
  assert(COLOR_CHIPS.filter((c) => c.id !== 'pick').length === 10, 'ten named color buckets');
}

function loadJpegRgba(file) {
  // Use sharp if present, else ffmpeg to raw, else fail clearly.
  return import('node:child_process').then(async ({ execFileSync }) => {
    const { tmpdir } = await import('node:os');
    const out = path.join(tmpdir(), `sock-color-${process.pid}.raw`);
    const info = execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', file], { encoding: 'utf8' }).trim();
    const [w, h] = info.split('x').map(Number);
    execFileSync('ffmpeg', ['-y', '-i', file, '-f', 'rawvideo', '-pix_fmt', 'rgba', out], { stdio: 'pipe' });
    const buf = fs.readFileSync(out);
    fs.unlinkSync(out);
    return { data: new Uint8ClampedArray(buf), w, h };
  });
}

const FRAME = '/workspace/sock-matcher-feedback/v1420/f03.jpg';
if (fs.existsSync(FRAME)) {
  const full = await loadJpegRgba(FRAME);
  // Crop the camera stage out of the phone screenshot (y ~ 108–624 on 360x779).
  const y0 = Math.round(full.h * 0.14), y1 = Math.round(full.h * 0.80);
  const cw = full.w, ch = y1 - y0;
  const crop = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    crop.set(full.data.subarray(((y0 + y) * full.w) * 4, ((y0 + y) * full.w + cw) * 4), y * cw * 4);
  }
  // Downscale to ~320px long side, like the live battery-saver buffer.
  const scale = 320 / Math.max(cw, ch);
  const ww = Math.max(8, Math.round(cw * scale)), hh = Math.max(8, Math.round(ch * scale));
  const small = new Uint8ClampedArray(ww * hh * 4);
  for (let y = 0; y < hh; y++) {
    const sy = Math.min(ch - 1, Math.round(y * ch / hh));
    for (let x = 0; x < ww; x++) {
      const sx = Math.min(cw - 1, Math.round(x * cw / ww));
      const si = (sy * cw + sx) * 4, di = (y * ww + x) * 4;
      small[di] = crop[si]; small[di + 1] = crop[si + 1]; small[di + 2] = crop[si + 2]; small[di + 3] = 255;
    }
  }
  const engine = createEngine();
  const counts = {};
  let maxN = 0, totalMs = 0;
  for (const id of ['pink', 'white', 'grey', 'black', 'blue', 'green', 'purple', 'red', 'yellow', 'multi']) {
    const t0 = performance.now();
    const roi = { y0: Math.round(hh * 0.12), y1: Math.round(hh * 0.88) };
    const res = engine.process(small, ww, hh, { mode: 'colors', colorQuery: { id }, still: true, sensitivity: 0.5, roi });
    const ms = performance.now() - t0;
    totalMs += ms;
    const n = (res.colorHits || []).length;
    counts[id] = n;
    maxN = Math.max(maxN, n);
    assert(n <= 400, `${id}: crazy over-segmentation (${n})`);
    console.log(`  ${id}: ${n} hits in ${ms.toFixed(1)}ms`);
  }
  assert(counts.white >= 8, `v1420 standing pile: many white socks, got ${counts.white}`);
  assert(counts.pink >= 3, `v1420 standing pile: several pink socks, got ${counts.pink}`);
  assert(counts.grey >= 3, `v1420 standing pile: several grey socks, got ${counts.grey}`);
  assert(maxN > 3, `no 1–3 cap: the busiest color has ${maxN} hits`);
  const t0 = performance.now();
  engine.process(small, ww, hh, { mode: 'colors', colorQuery: { id: 'white' }, still: true, sensitivity: 0.5, roi: { y0: Math.round(hh * 0.12), y1: Math.round(hh * 0.88) } });
  const ms = performance.now() - t0;
  assert(ms < 80, `color sort on a 320px frame is cheap (${ms.toFixed(1)}ms)`);
  console.log('v1420 counts', counts, `work ${ww}x${hh}`);
} else {
  console.log('skip: private v1420 frames are not on this machine');
}

if (failed) {
  console.log(`\n${failed} failed`);
  process.exit(1);
}
console.log('\nAll color-sort checks passed');
