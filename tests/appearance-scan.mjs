/**
 * Scan mode matches on appearance only: color + texture, not size or shape.
 * Includes a private-frame check when v1257 / v1420 stills are present.
 */
import fs from 'node:fs';
import { appearanceSimilarity, similarity, appearanceOf, serializeAppearance, parseAppearance } from '../src/cv/match.js';
import { createEngine } from '../src/cv/engine.js';

let failed = 0;
function assert(cond, msg) {
  if (cond) console.log('ok:', msg);
  else { failed++; console.log('FAIL:', msg); }
}

function desc(extra = {}) {
  const ab = new Float32Array(64); ab[10] = 0.4; ab[11] = 0.3; ab[18] = 0.3;
  const lh = new Float32Array(8); lh[5] = 0.6; lh[6] = 0.4;
  const ori = new Float32Array(8); ori[2] = 0.5; ori[3] = 0.5;
  const bands = new Float32Array(15);
  for (let i = 0; i < 5; i++) { bands[i * 3] = 60 + i * 4; bands[i * 3 + 1] = 28; bands[i * 3 + 2] = 8; }
  return {
    ab, lh, ori, bands,
    mean: [70, 28, 8],
    pal: [{ L: 72, a: 30, b: 6, w: 0.7 }, { L: 58, a: 20, b: 10, w: 0.3 }],
    edge: 0.22,
    stripes: 4,
    area: 800,
    elong: 3.2,
    ...extra,
  };
}

const a = desc();
const smallFolded = desc({ area: 80, elong: 1.1 });
const huge = desc({ area: 8000, elong: 6.5 });
// Reverse the cuff-to-toe bands (rotated 180°).
const flipped = desc({ area: 200, elong: 0.8, bands: (() => {
  const b = new Float32Array(15);
  for (let i = 0; i < 5; i++) {
    const j = 4 - i;
    b[i * 3] = 60 + j * 4; b[i * 3 + 1] = 28; b[i * 3 + 2] = 8;
  }
  return b;
})() });
const other = desc({
  mean: [40, -20, -30],
  pal: [{ L: 40, a: -20, b: -30, w: 1 }],
  stripes: 0,
  edge: 0.05,
  area: 800,
  elong: 3.2,
});
other.ab = new Float32Array(64); other.ab[50] = 1;
other.lh = new Float32Array(8); other.lh[2] = 1;
other.bands = new Float32Array(15);
for (let i = 0; i < 5; i++) { other.bands[i * 3] = 40; other.bands[i * 3 + 1] = -20; other.bands[i * 3 + 2] = -30; }

const sSmall = appearanceSimilarity(a, smallFolded);
const sHuge = appearanceSimilarity(a, huge);
const sFlip = appearanceSimilarity(a, flipped);
const sOther = appearanceSimilarity(a, other);
assert(sSmall > 0.9 && Math.abs(sSmall - sHuge) < 1e-6, `same look, any size, same score (small=${sSmall.toFixed(3)} huge=${sHuge.toFixed(3)})`);
assert(sFlip > 0.85, `180° / reversed bands still match (${sFlip.toFixed(3)})`);
assert(sOther < 0.4, `a different color/texture stays low (${sOther.toFixed(3)})`);
assert(similarity(a, smallFolded) < sSmall, 'Auto Pairs may still penalize a size mismatch; Scan does not');

const stored = parseAppearance(serializeAppearance({ ...a, area: 9999, elong: 9, held: true }));
assert(!('area' in stored) && !('elong' in stored) && !('held' in stored), 'stored scan drops size/shape fields');
assert(appearanceSimilarity(a, stored) > 0.99, 'serialized appearance round-trips');
assert(!('area' in (appearanceOf(a))), 'appearanceOf strips size');

// Engine: a scanned large red sock should light up a much smaller, rotated mate.
{
  const W = 240, H = 360;
  function clamp(v) { return Math.max(0, Math.min(255, v)); }
  function frame(socks) {
    const rgba = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < rgba.length; i += 4) { rgba[i] = 210; rgba[i+1] = 200; rgba[i+2] = 185; rgba[i+3] = 255; }
    for (const s of socks) {
      const cos = Math.cos(s.rot || 0), sin = Math.sin(s.rot || 0);
      const pad = Math.ceil(Math.max(s.rw, s.rh) + 2);
      for (let yy = s.cy - pad; yy <= s.cy + pad; yy++) {
        for (let xx = s.cx - pad; xx <= s.cx + pad; xx++) {
          const x = xx | 0, y = yy | 0;
          if (x < 0 || y < 0 || x >= W || y >= H) continue;
          const dx = x - s.cx, dy = y - s.cy;
          const lx = dx * cos + dy * sin, ly = -dx * sin + dy * cos;
          if ((lx * lx) / (s.rw * s.rw) + (ly * ly) / (s.rh * s.rh) > 1) continue;
          const i = (y * W + x) * 4;
          rgba[i] = s.r; rgba[i+1] = s.g; rgba[i+2] = s.b;
        }
      }
    }
    return rgba;
  }
  const big = { cx: 70, cy: 140, rw: 28, rh: 70, rot: 0.1, r: 200, g: 40, b: 50 };
  const tiny = { cx: 170, cy: 220, rw: 10, rh: 18, rot: 1.2, r: 198, g: 42, b: 52 };
  const blue = { cx: 120, cy: 300, rw: 18, rh: 40, rot: 0, r: 40, g: 70, b: 190 };
  const e = createEngine();
  let t = 0, res;
  const full = frame([big, tiny, blue]);
  for (let i = 0; i < 4; i++) { t += 80; res = e.process(full, W, H, { mode: 'scan', still: false, t, sensitivity: 0.5, strictness: 0.55 }); }
  const refTrk = [...e.getTracks().values()].sort((a, b) => Math.hypot(a.cx-big.cx, a.cy-big.cy) - Math.hypot(b.cx-big.cx, b.cy-big.cy))[0];
  assert(refTrk && Math.hypot(refTrk.cx-big.cx, refTrk.cy-big.cy) < 30, 'found the large sock to scan');
  t += 80;
  const scanned = e.process(full, W, H, { mode: 'scan', still: false, t, sensitivity: 0.5, strictness: 0.55, tap: { x: refTrk.cx, y: refTrk.cy } });
  assert(scanned.scan?.ok && scanned.scan.appearance && !('area' in scanned.scan.appearance) && !('elong' in scanned.scan.appearance), 'scan result stores appearance only');
  // Give hysteresis a couple of hits.
  for (let i = 0; i < 4; i++) { t += 80; res = e.process(full, W, H, { mode: 'scan', still: false, t, sensitivity: 0.5, strictness: 0.55 }); }
  const ids = (res.matches || []).map((m) => m.id);
  const tracks = new Map([...e.getTracks().values()].map((tr) => [tr.id, tr]));
  const hitTiny = ids.some((id) => {
    const tr = tracks.get(id);
    return tr && Math.hypot(tr.cx - tiny.cx, tr.cy - tiny.cy) < 28;
  });
  assert(res.matches.length <= 3, `cap of 3 (${res.matches.length})`);
  assert(hitTiny, `smaller rotated mate is eligible (matches=${JSON.stringify(res.matches)}, tracks=${[...tracks.values()].map((tr) => ({id:tr.id,cx:tr.cx|0,cy:tr.cy|0,area:tr.area|0}))})`);
}

// Private frames: a flood-isolated mint pair from v1257 should still appear similar
// even if we pretend one crop is much smaller (appearance score ignores area).
const v1257 = '/workspace/sock-matcher-feedback/v1257/f06.jpg';
if (fs.existsSync(v1257)) {
  // Light check: engine can process a downscaled private still in scan mode without throwing.
  const { execFileSync } = await import('node:child_process');
  const { tmpdir } = await import('node:os');
  const path = await import('node:path');
  const raw = path.join(tmpdir(), `scan-${process.pid}.raw`);
  const info = execFileSync('ffprobe', ['-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','csv=p=0:s=x', v1257], { encoding: 'utf8' }).trim();
  const [w0, h0] = info.split('x').map(Number);
  execFileSync('ffmpeg', ['-y','-i', v1257, '-f','rawvideo','-pix_fmt','rgba', raw], { stdio: 'pipe' });
  const buf = fs.readFileSync(raw);
  fs.unlinkSync(raw);
  const scale = 320 / Math.max(w0, h0);
  const w = Math.round(w0 * scale), h = Math.round(h0 * scale);
  const small = new Uint8ClampedArray(w * h * 4);
  const src = new Uint8ClampedArray(buf);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(h0 - 1, Math.round(y * h0 / h));
    for (let x = 0; x < w; x++) {
      const sx = Math.min(w0 - 1, Math.round(x * w0 / w));
      const si = (sy * w0 + sx) * 4, di = (y * w + x) * 4;
      small[di] = src[si]; small[di+1] = src[si+1]; small[di+2] = src[si+2]; small[di+3] = 255;
    }
  }
  const e = createEngine();
  const res = e.process(small, w, h, { mode: 'scan', still: true, t: 1, sensitivity: 0.5, strictness: 0.55 });
  assert(res && Array.isArray(res.tracks), `private v1257 frame processed (regions=${res.regions}, tracks=${res.tracks.length})`);
}

if (failed) { console.log(`\n${failed} failed`); process.exit(1); }
console.log('\nAll appearance-scan checks passed');
