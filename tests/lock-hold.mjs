/**
 * Automated lock-persistence checks for the live-camera path.
 * Synthetic sock frames, no browser, no private photos.
 */
import { createEngine } from '../src/cv/engine.js';

const W = 240, H = 360;
const DT = 66; // ~15 Hz
let failed = 0;

function assert(cond, msg) {
  if (!cond) { failed++; console.error('FAIL:', msg); }
  else console.log('ok:', msg);
}

function makeFrame(socks, light = 1) {
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const n = ((x * 17 + y * 31) % 7) - 3;
      rgba[i] = clamp((210 + n) * light);
      rgba[i + 1] = clamp((200 + n) * light);
      rgba[i + 2] = clamp((185 + n) * light);
      rgba[i + 3] = 255;
    }
  }
  for (const s of socks) drawSock(rgba, s, light);
  return rgba;
}

function clamp(v) { return Math.max(0, Math.min(255, v)); }

function drawSock(rgba, s, light) {
  const { cx, cy, rw, rh, rot = 0, r, g, b } = s;
  const cos = Math.cos(rot), sin = Math.sin(rot);
  const pad = Math.ceil(Math.max(rw, rh) + 2);
  for (let yy = cy - pad; yy <= cy + pad; yy++) {
    for (let xx = cx - pad; xx <= cx + pad; xx++) {
      const x = xx | 0, y = yy | 0;
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      const dx = x - cx, dy = y - cy;
      const lx = dx * cos + dy * sin, ly = -dx * sin + dy * cos;
      if ((lx * lx) / (rw * rw) + (ly * ly) / (rh * rh) > 1) continue;
      const cuff = ly > rh * 0.45;
      const i = (y * W + x) * 4;
      rgba[i] = clamp((cuff ? r * 0.72 : r) * light);
      rgba[i + 1] = clamp((cuff ? g * 0.72 : g) * light);
      rgba[i + 2] = clamp((cuff ? b * 0.72 : b) * light);
    }
  }
}

const RED_A = { cx: 64, cy: 150, rw: 22, rh: 62, rot: 0.15, r: 200, g: 40, b: 50 };
const RED_B = { cx: 176, cy: 168, rw: 21, rh: 60, rot: -0.2, r: 198, g: 44, b: 48 };
const BLUE = { cx: 120, cy: 280, rw: 22, rh: 58, rot: 0.05, r: 40, g: 70, b: 190 };

function step(e, rgba, t, extra = {}) {
  return e.process(rgba, W, H, { mode: 'scan', still: false, t, sensitivity: 0.5, strictness: 0.55, ...extra });
}

function nearest(e, sock) {
  let best = null, bd = 1e9;
  for (const t of e.getTracks().values()) {
    const d = Math.hypot(t.cx - sock.cx, t.cy - sock.cy);
    if (d < bd) { bd = d; best = t; }
  }
  return bd < 28 ? best : null;
}

// --- 1. Lock survives ~1.5s of misses, then drops after ~2.5s ---
{
  const e = createEngine();
  const full = makeFrame([RED_A, RED_B, BLUE]);
  const missing = makeFrame([RED_A, BLUE]);
  let t = 0, r;
  for (let i = 0; i < 5; i++) { t += DT; r = step(e, full, t); }
  const refTrk = nearest(e, RED_A);
  assert(refTrk, 'detected the scanned sock');
  t += DT; r = step(e, full, t, { tap: { x: refTrk.cx, y: refTrk.cy } });
  assert(r.scan?.ok, `tap set a reference (${r.scan && r.scan.ok})`);
  for (let i = 0; i < 6; i++) { t += DT; r = step(e, full, t); }
  assert(r.matches.length >= 1, `acquired a live match (n=${r.matches.length})`);
  assert(r.matches.length <= 3, `cap of 3 matches (n=${r.matches.length})`);
  const mate = nearest(e, RED_B);
  const locked = mate && r.matches.some((m) => m.id === mate.id);
  assert(locked, `locked onto the mate sock (ids=${r.matches.map((m) => m.id).join(',')}, mate=${mate && mate.id})`);
  const lockedId = locked ? mate.id : (r.matches[0] && r.matches[0].id);

  let stillShownEarly = false, droppedBy1s = false;
  const tMiss0 = t;
  for (let i = 0; i < 30; i++) {
    t += DT; r = step(e, missing, t);
    const shown = r.matches.some((m) => m.id === lockedId);
    const elapsed = t - tMiss0;
    if (elapsed >= 150 && elapsed <= 280 && shown) stillShownEarly = true;
    if (elapsed >= 900 && !shown) droppedBy1s = true;
  }
  assert(stillShownEarly, 'lock still shown after a brief miss (~0.2s)');
  assert(droppedBy1s, 'lock dropped after ~0.9s of misses (light hysteresis only)');
}

// --- 2. Same track id through brief misses, small motion, and a lighting dip ---
{
  const e = createEngine();
  let t = 0, r;
  const base = [RED_A, RED_B, BLUE];
  for (let i = 0; i < 5; i++) { t += DT; r = step(e, makeFrame(base), t); }
  const refTrk = nearest(e, RED_A);
  t += DT; r = step(e, makeFrame(base), t, { tap: { x: refTrk.cx, y: refTrk.cy } });
  for (let i = 0; i < 6; i++) { t += DT; r = step(e, makeFrame(base), t); }
  const mate = nearest(e, RED_B);
  assert(mate && r.matches.some((m) => m.id === mate.id), 'motion test acquired the mate');
  const id = mate && mate.id;

  // Two missed frames (mate gone), then it returns nearby.
  t += DT; r = step(e, makeFrame([RED_A, BLUE]), t);
  t += DT; r = step(e, makeFrame([RED_A, BLUE]), t);
  const afterMiss = e.getTracks().get(id);
  assert(afterMiss && afterMiss.missed >= 1, `track ${id} coasted through 2 misses (missed=${afterMiss && afterMiss.missed})`);

  // Reappear, slightly shifted, then drift a few pixels per frame.
  for (let i = 0; i < 8; i++) {
    const moved = { ...RED_B, cx: RED_B.cx + 3 * i, cy: RED_B.cy + i };
    t += DT; r = step(e, makeFrame([RED_A, moved, BLUE]), t);
  }
  const movedTrk = e.getTracks().get(id);
  assert(movedTrk && movedTrk.missed === 0, `same track id ${id} re-associated after motion`);
  assert(r.matches.some((m) => m.id === id), 'highlight stayed on the same sock through motion');

  // Darker lighting for several frames.
  for (let i = 0; i < 6; i++) {
    t += DT; r = step(e, makeFrame(base, 0.78), t);
  }
  assert(r.matches.some((m) => m.id === id), 'highlight survived a lighting dip');
  assert(r.matches.length <= 3, 'still at most 3 matches');
}

// --- 3. Per-frame cost stays inside a 15 Hz budget on a 90k-px buffer ---
{
  const e = createEngine();
  const full = makeFrame([RED_A, RED_B, BLUE]);
  let t = 0;
  // warmup
  for (let i = 0; i < 4; i++) { t += DT; step(e, full, t); }
  const times = [];
  for (let i = 0; i < 20; i++) {
    t += DT;
    const t0 = performance.now();
    const r = step(e, full, t);
    times.push(performance.now() - t0);
    void r;
  }
  times.sort((a, b) => a - b);
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  const p95 = times[Math.min(times.length - 1, Math.floor(times.length * 0.95))];
  assert(avg < 50, `avg process ${avg.toFixed(1)}ms is under 50ms (~15 Hz budget)`);
  assert(p95 < 66, `p95 process ${p95.toFixed(1)}ms is under 66ms`);
  console.log(`perf: avg ${avg.toFixed(1)}ms p95 ${p95.toFixed(1)}ms`);
}

if (failed) {
  console.error(`\n${failed} assertion(s) failed`);
  process.exit(1);
}
console.log('\nAll lock-hold checks passed');
