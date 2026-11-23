import { createVision } from './vision.js';
import { createCamera } from './camera.js';
import { generateDemoPile } from './demo.js';
import { createOverlay, SLOT_COLORS, MATCH_COLOR } from './overlay.js';
import { createMlClient } from './ml/client.js';
import { COLOR_CHIPS, chipTint } from './cv/colorClass.js';
import { serializeAppearance, parseAppearance } from './cv/match.js';

const $ = (id) => document.getElementById(id);
const video = $('video');
const frame = $('frame');
const overlayCanvas = $('overlay');
const gate = $('gate');
const chip = $('status-chip');
const toastEl = $('toast');
const fileInput = $('file-input');
const shutter = $('btn-shutter');
const refCard = $('ref-card');
const refThumb = $('ref-thumb');
const hud = $('perf-hud');
const DEBUG = new URLSearchParams(location.search).has('debug');

const camera = createCamera(video);
const overlay = createOverlay(overlayCanvas);
const vision = createVision(onResult);
const ml = createMlClient();
const params = new URLSearchParams(location.search);
const DEMO_MODE = params.has('demo');
const USE_ML = !DEMO_MODE && !params.has('noml');

const MAX_VISION_SIDE = 320; // live camera work buffer, long side
const STILL_PIXELS = 140_000;
const MIN_INTERVAL_SAVER = 125; // ~8 Hz classic detector
const MIN_INTERVAL_PERF = 90;   // still conservative if saver is off
const ML_INTERVAL_SAVER = 700;  // ~1.4 Hz embeddings, a few crops only
const ML_INTERVAL_PERF = 500;   // ~2 Hz
const MAX_ML_CROPS = 6;
const SAVER_KEY = 'sock-matcher-battery-saver';
const SCAN_KEY = 'sock-matcher-scan';

function readBatterySaver() {
  try {
    const v = localStorage.getItem(SAVER_KEY);
    if (v == null) return true;
    return v !== '0';
  } catch { return true; }
}

let batterySaver = readBatterySaver();

function setBatterySaver(on) {
  batterySaver = !!on;
  try { localStorage.setItem(SAVER_KEY, on ? '1' : '0'); } catch { /* ignore */ }
}

function classicInterval() { return batterySaver ? MIN_INTERVAL_SAVER : MIN_INTERVAL_PERF; }
function mlInterval() { return batterySaver ? ML_INTERVAL_SAVER : ML_INTERVAL_PERF; }

const state = {
  mode: 'scan',
  source: null, // 'camera' | 'photo' | 'demo'
  still: null,
  frozen: false,
  torch: false,
  sensitivity: 0.5,
  strictness: 0.55,
  awaitingSurfaceTap: false,
  view: { w: 0, h: 0, dpr: 1 },
  pending: {},
  submissions: new Map(),
  result: null,
  resultMap: null,
  resultRect: null,
  hasRef: false,
  colorQuery: { id: 'pink' },
  mlResult: null,
  lastSubmit: 0,
  stillDirty: false,
  demoTruth: null,
  waiters: [],
};

const perf = {
  frames: [], // rAF intervals
  overlayMs: [],
  grabMs: [],
  visionMs: [],
  visionTimes: [],
  last: 0,
};

let toastTimer = 0;
function toast(msg, ms = 2200) {
  toastEl.hidden = false;
  toastEl.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.hidden = true; }, ms);
}

function setChip(text) {
  chip.hidden = !gate.classList.contains('hidden') || !text;
  chip.textContent = text || '';
  chip.classList.toggle('frozen', state.frozen);
}

function measure() {
  const stage = $('stage');
  const w = stage.clientWidth, h = stage.clientHeight;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  state.view = { w, h, dpr };
  for (const c of [overlayCanvas, frame]) {
    const cw = Math.round(w * dpr), ch = Math.round(h * dpr);
    if (c.width !== cw || c.height !== ch) { c.width = cw; c.height = ch; }
  }
  if (state.still) drawStill();
}

/** Where the source is shown on screen and which part of it we analyse. */
function layoutFor(src) {
  const { w: vw, h: vh } = state.view;
  const sw0 = src === video ? video.videoWidth : src.width;
  const sh0 = src === video ? video.videoHeight : src.height;
  if (!sw0 || !sh0 || !vw || !vh) return null;
  if (src === video) {
    // object-fit: cover — analyse exactly the visible part.
    const s = Math.max(vw / sw0, vh / sh0);
    const sw = vw / s, sh = vh / s;
    const rect = { sx: (sw0 - sw) / 2, sy: (sh0 - sh) / 2, sw, sh };
    let ww, wh;
    if (vw >= vh) {
      ww = MAX_VISION_SIDE;
      wh = Math.max(120, Math.round(MAX_VISION_SIDE * vh / vw));
    } else {
      wh = MAX_VISION_SIDE;
      ww = Math.max(120, Math.round(MAX_VISION_SIDE * vw / vh));
    }
    const k = vw / ww;
    const mirror = camera.isFront();
    return {
      rect, w: ww, h: wh,
      map: {
        scale: k, mirror,
        toView: (x, y) => [mirror ? vw - x * k : x * k, y * k],
        toWork: (x, y) => [(mirror ? vw - x : x) / k, y / k],
      },
    };
  }
  const s = Math.min(vw / sw0, vh / sh0);
  const dw = sw0 * s, dh = sh0 * s, ox = (vw - dw) / 2, oy = (vh - dh) / 2;
  const ww = Math.round(Math.sqrt(STILL_PIXELS * sw0 / sh0));
  const wh = Math.round(ww * sh0 / sw0);
  const k = dw / ww;
  return {
    rect: { sx: 0, sy: 0, sw: sw0, sh: sh0 }, w: ww, h: wh, display: { ox, oy, dw, dh },
    map: {
      scale: k, mirror: false,
      toView: (x, y) => [ox + x * k, oy + y * k],
      toWork: (x, y) => [(x - ox) / k, (y - oy) / k],
    },
  };
}

/** Work-space rows not hidden behind the top bar and the bottom dock. */
function roiFor(lay) {
  const top = $('topbar').getBoundingClientRect().bottom;
  const dock = $('dock').getBoundingClientRect().top;
  const y0 = lay.map.toWork(0, top)[1];
  const y1 = lay.map.toWork(0, dock - 8)[1];
  return { y0: Math.max(0, y0), y1: Math.min(lay.h, y1) };
}

function currentSource() {
  if (state.still) return state.still;
  if (state.source === 'camera' && video.videoWidth) return video;
  return null;
}

function drawStill() {
  const { w, h, dpr } = state.view;
  const fctx = frame.getContext('2d');
  fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  fctx.fillStyle = '#111';
  fctx.fillRect(0, 0, w, h);
  const lay = layoutFor(state.still);
  if (!lay) return;
  const { ox, oy, dw, dh } = lay.display;
  fctx.drawImage(state.still, ox, oy, dw, dh);
}

function submit(now) {
  const src = currentSource();
  if (!src) return false;
  const lay = layoutFor(src);
  if (!lay) return false;
  const opts = {
    mode: state.mode,
    sensitivity: state.sensitivity,
    strictness: state.strictness,
    still: !!state.still,
    t: now,
    roi: roiFor(lay),
    colorQuery: state.mode === 'colors' ? state.colorQuery : undefined,
    ...state.pending,
  };
  const g0 = performance.now();
  const seq = vision.submit(src, lay.rect, lay.w, lay.h, opts);
  if (!seq) return false;
  push(perf.grabMs, performance.now() - g0);
  state.pending = {};
  state.lastSubmit = now;
  state.submissions.set(seq, lay);
  if (state.submissions.size > 8) state.submissions.delete(state.submissions.keys().next().value);
  return true;
}

function hasPending() {
  const p = state.pending;
  return !!(p.scanBox || p.tap || p.colorTap || p.surfaceTap || p.surfaceAuto);
}

function queueVision() {
  if (state.still) {
    reprocessStill();
    return;
  }
  state.forceNext = true;
  if (!vision.isBusy()) {
    submit(performance.now());
    state.forceNext = false;
  }
}

const sigCanvas = document.createElement('canvas');
sigCanvas.width = 12;
sigCanvas.height = 12;
const sigCtx = sigCanvas.getContext('2d', { willReadFrequently: true });
let lastSig = null;

function barelyChanged() {
  if (hasPending()) return false;
  const src = currentSource();
  if (!src) return false;
  try {
    sigCtx.drawImage(src, 0, 0, 12, 12);
    const d = sigCtx.getImageData(0, 0, 12, 12).data;
    if (!lastSig) { lastSig = new Uint8Array(d); return false; }
    let acc = 0;
    for (let i = 0; i < d.length; i += 4) {
      acc += Math.abs(d[i] - lastSig[i]) + Math.abs(d[i + 1] - lastSig[i + 1]) + Math.abs(d[i + 2] - lastSig[i + 2]);
    }
    if (acc / 144 < 8) return true;
    lastSig = new Uint8Array(d);
    return false;
  } catch {
    return false;
  }
}

function onResult(res) {
  const lay = state.submissions.get(res.seq) || [...state.submissions.values()].pop();
  state.result = res;
  state.resultMap = lay.map;
  state.resultRect = lay.rect;
  state.resultLayout = lay;
  push(perf.visionMs, res.timing.total);
  perf.visionTimes.push(performance.now());
  if (perf.visionTimes.length > 30) perf.visionTimes.shift();
  if (res.scan) onScanResult(res, lay);
  if (res.colorPick) {
    state.colorQuery = {
      id: 'pick',
      L: res.colorPick.L,
      a: res.colorPick.a,
      b: res.colorPick.b,
      name: res.colorPick.name,
    };
    syncColorChips();
    toast(`Picked ${res.colorPick.name}`);
    // Hits on this frame were computed before the pick existed — run once more.
    queueVision();
  }
  state.hasRef = !!res.ref || state.hasRef;
  updateTargets(res);
  updateUi();
  maybeScore(res);
  if (state.stillDirty && state.still) { state.stillDirty = false; submit(performance.now()); }
  const waiters = state.waiters.splice(0);
  for (const fn of waiters) fn(res);
}

function displayLists(res) {
  const mlMatches = state.mlResult?.matches || [];
  const mlPairs = state.mlResult?.pairs || [];
  // Classic highlights stay up until the model actually returns confident matches.
  if (state.mode === 'scan') {
    if (mlMatches.length) return { matches: mlMatches, pairs: [] };
    return { matches: res.matches || [], pairs: [] };
  }
  if (mlPairs.length) return { matches: [], pairs: mlPairs };
  return { matches: [], pairs: res.pairs || [] };
}

function mlStatusText() {
  if (!USE_ML) return '';
  const s = ml.getState();
  if (s.status === 'loading') return `Loading smart matching… ${Math.max(0, Math.round(s.progress || 0))}%`;
  if (s.status === 'error') return 'Smart matching unavailable';
  return '';
}

function updateTargets(res) {
  if (!res) return;
  if (state.mode === 'colors') {
    const hits = res.colorHits || [];
    const tint = chipTint(state.colorQuery?.id || 'pink');
    const targets = hits.map((h, i) => ({
      key: `c${Math.round(h.cx / 6)}_${Math.round(h.cy / 6)}`,
      track: { id: i, cx: h.cx, cy: h.cy, vx: 0, vy: 0, outline: h.outline, box: h.box },
      color: tint,
      kind: 'color',
    }));
    overlay.setTargets(targets, performance.now());
    return;
  }
  const byId = new Map(res.tracks.map((t) => [t.id, t]));
  const lists = displayLists(res);
  const targets = [];
  if (state.mode === 'scan') {
    for (const m of lists.matches) {
      const t = byId.get(m.id);
      if (!t) continue;
      // Fresh scores may land before the next vision outline is attached.
      if (!t.outline && t.box) t.outline = boxOutline(t.box);
      if (!t.outline) continue;
      targets.push({ key: `m${m.id}`, track: t, color: MATCH_COLOR, kind: 'match', label: `${Math.round(m.score * 100)}%` });
    }
    const refT = res.ref?.excludeId && byId.get(res.ref.excludeId);
    if (refT && refT.outline) targets.push({ key: `r${refT.id}`, track: refT, kind: 'ref', color: '#ffffff' });
  } else {
    for (const p of lists.pairs) {
      const color = SLOT_COLORS[(p.slot - 1) % SLOT_COLORS.length];
      for (const id of [p.a, p.b]) {
        const t = byId.get(id);
        if (!t) continue;
        if (!t.outline && t.box) t.outline = boxOutline(t.box);
        if (!t.outline) continue;
        targets.push({ key: `p${p.a}-${p.b}-${id}`, track: t, color, kind: 'pair', label: String(p.slot) });
      }
    }
  }
  overlay.setTargets(targets, performance.now());
}

function boxOutline(box) {
  const [x0, y0, x1, y1] = box;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return [
    x0 - cx, y0 - cy, x1 - cx, y0 - cy,
    x1 - cx, y1 - cy, x0 - cx, y1 - cy,
  ];
}

function onScanResult(res, lay) {
  if (!res.scan.ok) {
    toast('Couldn’t find a sock there — try again');
    return;
  }
  // Thumbnail of the scanned sock, cropped from the source at full resolution.
  const src = currentSource();
  if (!src) return;
  const [x0, y0, x1, y1] = res.scan.box;
  const fx = lay.rect.sw / lay.w, fy = lay.rect.sh / lay.h;
  const pad = 3;
  let sx = lay.rect.sx + (x0 - pad) * fx, sy = lay.rect.sy + (y0 - pad) * fy;
  let sw = (x1 - x0 + 2 * pad) * fx, sh = (y1 - y0 + 2 * pad) * fy;
  const side = Math.max(sw, sh);
  sx -= (side - sw) / 2; sy -= (side - sh) / 2;
  const tctx = refThumb.getContext('2d');
  tctx.fillStyle = '#222';
  tctx.fillRect(0, 0, refThumb.width, refThumb.height);
  try { tctx.drawImage(src, sx, sy, side, side, 0, 0, refThumb.width, refThumb.height); } catch { /* ignore */ }
  refCard.hidden = false;
  if (state.mode !== 'scan') setMode('scan', true);
  toast('Scanned — now point at the pile');
  if (res.scan.appearance) persistScan({ desc: res.scan.appearance, excludeId: res.scan.id });
  requestRefEmbed(res, lay);
}

function colorCountLabel(n) {
  const q = state.colorQuery || { id: 'pink' };
  const chip = COLOR_CHIPS.find((c) => c.id === q.id);
  const name = q.id === 'pick'
    ? (q.name ? `that ${q.name}` : 'that color')
    : (chip?.label || q.id).toLowerCase();
  if (q.id === 'multi') return n ? `${n} patterned sock${n === 1 ? '' : 's'}` : 'Looking for patterned socks…';
  return n ? `${n} ${name} sock${n === 1 ? '' : 's'}` : `Looking for ${name} socks…`;
}

function updateUi() {
  const res = state.result;
  if (state.mode === 'colors') {
    const n = res?.colorHits?.length || 0;
    const picking = state.colorQuery?.id === 'pick' && !Number.isFinite(state.colorQuery?.L);
    setChip(picking ? 'Tap a sock to pick its color' : colorCountLabel(n));
    refCard.hidden = true;
    return;
  }
  if (state.mode === 'scan') {
    shutter.querySelector('.shutter-label').textContent = state.hasRef ? 'Rescan' : 'Scan';
    shutter.classList.remove('on');
    if (!state.hasRef) setChip(state.still ? 'Tap a sock or Scan the one in the box' : 'Tap a sock, or Scan the one in the box');
    else {
      const lists = res ? displayLists(res) : { matches: [] };
      if (lists.matches.length) setChip(`Best match ${Math.round(lists.matches[0].score * 100)}%`);
      else setChip(mlStatusText() || 'Looking for its match…');
    }
  } else {
    shutter.querySelector('.shutter-label').textContent = state.frozen ? 'Resume' : 'Freeze';
    shutter.classList.toggle('on', state.frozen);
    const n = res ? displayLists(res).pairs.length : 0;
    const loading = mlStatusText();
    setChip(`${state.frozen ? 'Frozen · ' : ''}${n ? `${n} likely pair${n === 1 ? '' : 's'}` : (loading || 'Looking for pairs…')}`);
  }
  refCard.hidden = !state.hasRef;
}

function targetBox() {
  const { w, h } = state.view;
  const side = Math.min(w * 0.5, h * 0.3);
  return { x: (w - side) / 2, y: h * 0.42 - side / 2, w: side, h: side };
}

function loop(ts) {
  requestAnimationFrame(loop);
  if (document.hidden) return;
  const dt = perf.last ? ts - perf.last : 16.7;
  perf.last = ts;
  push(perf.frames, dt, 240);
  const live = state.source === 'camera' && !state.frozen && video.videoWidth;
  const due = state.forceNext || (ts - state.lastSubmit >= classicInterval());
  if (live && due && !vision.isBusy()) {
    const must = state.forceNext || hasPending();
    if (must || !barelyChanged()) {
      submit(ts);
      state.forceNext = false;
    } else {
      state.lastSubmit = ts;
    }
  }
  const showTarget = state.mode === 'scan' && !state.hasRef && state.source && gate.classList.contains('hidden');
  const map = state.resultMap || { scale: 1, mirror: false, toView: (x, y) => [x, y] };
  const ms = overlay.render(ts, Math.min(100, dt), state.view, map, { target: showTarget ? targetBox() : null });
  if (ms) push(perf.overlayMs, ms);
  if (DEBUG && ts - (perf.hudT || 0) > 500) { perf.hudT = ts; hud.hidden = false; hud.textContent = perfText(); }
}

function push(arr, v, max = 120) { arr.push(v); if (arr.length > max) arr.shift(); }
function stats(arr) {
  if (!arr.length) return { avg: 0, p95: 0 };
  const s = arr.slice().sort((a, b) => a - b);
  return { avg: s.reduce((a, b) => a + b, 0) / s.length, p95: s[Math.floor(s.length * 0.95)] };
}
function perfSummary() {
  const f = stats(perf.frames);
  const t = perf.visionTimes;
  return {
    fps: f.avg ? 1000 / f.avg : 0,
    frameAvg: f.avg, frameP95: f.p95,
    jank: perf.frames.filter((d) => d > 34).length,
    overlay: stats(perf.overlayMs), grab: stats(perf.grabMs), vision: stats(perf.visionMs),
    visionHz: t.length > 1 ? (t.length - 1) * 1000 / (t[t.length - 1] - t[0]) : 0,
    engine: vision.kind(),
  };
}
function perfText() {
  const p = perfSummary();
  return `ui ${p.fps.toFixed(0)}fps p95 ${p.frameP95.toFixed(1)}ms\n`
    + `vision ${p.visionHz.toFixed(1)}Hz ${p.vision.avg.toFixed(1)}ms (${p.engine})\n`
    + `grab ${p.grab.avg.toFixed(1)}ms overlay ${p.overlay.avg.toFixed(2)}ms`;
}


function persistScan(patch) {
  try {
    const cur = JSON.parse(sessionStorage.getItem(SCAN_KEY) || '{}');
    if (patch.desc) patch.desc = serializeAppearance(patch.desc);
    if (patch.embedding && patch.embedding.length) patch.embedding = Array.from(patch.embedding);
    let thumb = cur.thumb;
    try { thumb = refThumb.toDataURL('image/jpeg', 0.72); } catch { /* ignore */ }
    sessionStorage.setItem(SCAN_KEY, JSON.stringify({ ...cur, ...patch, thumb, savedAt: Date.now() }));
  } catch { /* private mode / quota */ }
}

function forgetScan() {
  try { sessionStorage.removeItem(SCAN_KEY); } catch { /* ignore */ }
}

function readSavedScan() {
  try {
    const raw = sessionStorage.getItem(SCAN_KEY);
    if (!raw) return null;
    const o = JSON.parse(raw);
    return o && o.desc ? o : null;
  } catch { return null; }
}

async function ensureScanInEngine() {
  const saved = readSavedScan();
  if (!saved) return false;
  const cur = await vision.getAppearance();
  if (!cur || !cur.desc) {
    const desc = parseAppearance(saved.desc);
    if (desc) await vision.setAppearance(desc, saved.excludeId);
  }
  if (saved.embedding && !ml.getState()?.error) {
    startMl();
    await ml.setRefEmbedding(saved.embedding);
  }
  if (saved.thumb && !state.hasRef) {
    try {
      const img = new Image();
      await new Promise((resolve) => { img.onload = resolve; img.onerror = resolve; img.src = saved.thumb; });
      if (img.width) {
        const ctx = refThumb.getContext('2d');
        ctx.fillStyle = '#222';
        ctx.fillRect(0, 0, refThumb.width, refThumb.height);
        ctx.drawImage(img, 0, 0, refThumb.width, refThumb.height);
      }
    } catch { /* ignore */ }
  }
  state.hasRef = true;
  refCard.hidden = false;
  updateUi();
  return true;
}

function resetVision(clearRef = false) {
  vision.reset(!!state.still, clearRef);
  lastSig = null;
  overlay.clearAll();
  state.result = null;
  state.mlResult = null;
  if (clearRef) { state.hasRef = false; refCard.hidden = true; ml.clearRef(); }
}

function reprocessStill() {
  if (!state.still) return;
  if (!submit(performance.now())) state.stillDirty = true;
}

async function startCamera() {
  try {
    state.still = null;
    state.source = 'camera';
    state.demoTruth = null;
    frame.classList.remove('show');
    resetVision(false);
    if (!batterySaver) startMl();
    await camera.start('environment');
    await ensureScanInEngine();
    video.classList.toggle('mirror', camera.isFront());
    syncTorch();
    setFrozen(false);
    gate.classList.add('hidden');
    updateUi();
    toast('Put one sock in the box and tap Scan');
  } catch (err) {
    console.warn(err);
    toast('Camera blocked — try Demo or Upload a photo');
  }
}

function useStill(canvas, source, truth = null) {
  if (source === 'photo' && !batterySaver) startMl();
  camera.stop();
  syncTorch();
  state.still = canvas;
  state.source = source;
  state.demoTruth = truth;
  state.frozen = false;
  frame.classList.add('show');
  gate.classList.add('hidden');
  resetVision(false);
  measure();
  updateUi();
  reprocessStill();
  ensureScanInEngine();
}

function loadDemo() {
  const demo = generateDemoPile();
  useStill(demo.canvas, 'demo', demo.socks);
  toast('Demo pile — tap a sock to find its match');
}

function loadImageFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas');
    const scale = Math.min(1, 1600 / Math.max(img.width, img.height));
    c.width = Math.round(img.width * scale);
    c.height = Math.round(img.height * scale);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(url);
    useStill(c, 'photo');
  };
  img.onerror = () => { URL.revokeObjectURL(url); toast('Could not read that photo'); };
  img.src = url;
}

function setFrozen(on) {
  state.frozen = on;
  if (on && state.source === 'camera') {
    const { w, h, dpr } = state.view;
    const fctx = frame.getContext('2d');
    fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const s = Math.max(w / video.videoWidth, h / video.videoHeight);
    const dw = video.videoWidth * s, dh = video.videoHeight * s;
    fctx.save();
    if (camera.isFront()) { fctx.translate(w, 0); fctx.scale(-1, 1); }
    fctx.drawImage(video, (w - dw) / 2, (h - dh) / 2, dw, dh);
    fctx.restore();
    frame.classList.add('show');
  } else if (state.source === 'camera') {
    frame.classList.remove('show');
  }
  updateUi();
}

function setMode(mode, keepRef = false) {
  if (state.mode === mode) return;
  state.mode = mode;
  $('mode-scan').classList.toggle('active', mode === 'scan');
  $('mode-pairs').classList.toggle('active', mode === 'pairs');
  $('mode-colors').classList.toggle('active', mode === 'colors');
  $('mode-scan').setAttribute('aria-selected', String(mode === 'scan'));
  $('mode-pairs').setAttribute('aria-selected', String(mode === 'pairs'));
  $('mode-colors').setAttribute('aria-selected', String(mode === 'colors'));
  document.getElementById('app').classList.toggle('mode-colors', mode === 'colors');
  const chips = $('color-chips');
  if (chips) chips.hidden = mode !== 'colors';
  shutter.hidden = mode === 'colors';
  if (state.frozen && mode === 'scan') setFrozen(false);
  overlay.setTargets([], performance.now());
  if (mode === 'colors') syncColorChips();
  updateUi();
  reprocessStill();
  // Live camera: force a frame so color hits appear right away.
  if (!state.still && state.source === 'camera') queueVision();
}

function workSnapshot(lay) {
  const src = currentSource();
  if (!src || !lay) return null;
  return vision.snapshot(src, lay.rect, lay.w, lay.h);
}

function paintThumb(box, lay) {
  const src = currentSource();
  if (!src || !box || !lay) return;
  const [x0, y0, x1, y1] = box;
  const fx = lay.rect.sw / lay.w, fy = lay.rect.sh / lay.h;
  const pad = 3;
  let sx = lay.rect.sx + (x0 - pad) * fx, sy = lay.rect.sy + (y0 - pad) * fy;
  let sw = (x1 - x0 + 2 * pad) * fx, sh = (y1 - y0 + 2 * pad) * fy;
  const side = Math.max(sw, sh);
  sx -= (side - sw) / 2; sy -= (side - sh) / 2;
  const tctx = refThumb.getContext('2d');
  tctx.fillStyle = '#222';
  tctx.fillRect(0, 0, refThumb.width, refThumb.height);
  try { tctx.drawImage(src, sx, sy, side, side, 0, 0, refThumb.width, refThumb.height); } catch { /* ignore */ }
}

async function applyPick(pick, lay) {
  if (!pick?.ok || !pick.mask) return false;
  const applied = await vision.setRefMask(pick.mask);
  if (applied && applied.ok === false) return false;
  if (pick.embedding) await ml.setRefEmbedding(pick.embedding);
  state.hasRef = true;
  state.mlResult = null;
  paintThumb(pick.box, lay);
  refCard.hidden = false;
  if (state.mode !== 'scan') setMode('scan', true);
  toast('Scanned — now point at the pile');
  updateUi();
  reprocessStill();
  return true;
}

function startMl() {
  if (!USE_ML) return;
  ml.start();
}

ml.onChange((s) => {
  const el = $('model-progress');
  if (!el) return;
  if (s.status === 'loading') {
    el.hidden = false;
    const bar = $('model-progress-bar');
    const label = $('model-progress-label');
    if (bar) bar.style.width = `${Math.max(2, s.progress || 0)}%`;
    if (label) label.textContent = `Loading smart matching… ${Math.max(0, Math.round(s.progress || 0))}%`;
  } else if (s.status === 'ready') {
    el.hidden = true;
    if (state.result) maybeScore(state.result);
    updateUi();
  } else if (s.status === 'error') {
    el.hidden = true;
    toast('Smart matching unavailable');
    updateUi();
  }
});

async function maybeScore(res) {
  if (!USE_ML || !ml.isReady() || !res || document.hidden) return;
  if (state.mode === 'colors') return;
  if (state.mode === 'scan' && !state.hasRef) return;
  const lay = state.resultLayout;
  const img = workSnapshot(lay);
  if (!img) return;
  const tracks = (res.tracks || []).filter((t) => t.box && (t.missed || 0) === 0);
  const out = await ml.score(img.data, img.width, img.height, tracks, {
    mode: state.mode,
    now: performance.now(),
    minInterval: mlInterval(),
    maxCrops: MAX_ML_CROPS,
  });
  if (!out) return;
  state.mlResult = out;
  updateTargets(state.result);
  updateUi();
}

function requestRefEmbed(res, lay) {
  if (!USE_ML || !res?.scan?.ok) return;
  startMl();
  const box = res.scan.box
    ? { x0: res.scan.box[0], y0: res.scan.box[1], x1: res.scan.box[2], y1: res.scan.box[3] }
    : null;
  waitMlReady().then(async (ok) => {
    if (!ok) return;
    const img = workSnapshot(state.resultLayout || lay);
    if (!img || !box) return;
    const out = await ml.embedRef(img.data, img.width, img.height, { box });
    if (out?.ok && out.embedding) persistScan({ embedding: out.embedding });
    if (out?.ok && state.result) maybeScore(state.result);
  });
}

function waitMlReady(ms = 120_000) {
  if (ml.isReady()) return Promise.resolve(true);
  if (ml.getState().status === 'error') return Promise.resolve(false);
  return new Promise((resolve) => {
    const t = setTimeout(() => { un(); resolve(ml.isReady()); }, ms);
    const un = ml.onChange((s) => {
      if (s.status === 'ready' || s.status === 'error') {
        clearTimeout(t);
        un();
        resolve(s.status === 'ready');
      }
    });
  });
}

async function doScan() {
  const lay = state.resultLayout || layoutFor(currentSource() || video);
  if (!lay) { toast('Start the camera first'); return; }
  const b = targetBox();
  const [x0, y0] = lay.map.toWork(b.x, b.y);
  const [x1, y1] = lay.map.toWork(b.x + b.w, b.y + b.h);
  const box = { x0: Math.min(x0, x1), y0, x1: Math.max(x0, x1), y1 };
  if (state.frozen) setFrozen(false);
  // Classic scan always runs immediately. The model only embeds afterwards.
  state.pending.scanBox = box;
  queueVision();
}

function clearRef() {
  vision.clearRef();
  ml.clearRef();
  forgetScan();
  state.hasRef = false;
  state.mlResult = null;
  refCard.hidden = true;
  overlay.setTargets([], performance.now());
  updateUi();
  reprocessStill();
}

function onTap(ev) {
  if (!gate.classList.contains('hidden')) return;
  const lay = state.resultLayout;
  if (!lay) return;
  const rect = overlayCanvas.getBoundingClientRect();
  const [wx, wy] = lay.map.toWork(ev.clientX - rect.left, ev.clientY - rect.top);
  if (state.awaitingSurfaceTap) {
    state.awaitingSurfaceTap = false;
    state.pending.surfaceTap = { x: wx, y: wy };
    $('surface-note').textContent = 'Surface color is locked. Tap Auto Surface to estimate it again.';
    overlay.clearAll();
    toast('Surface locked');
    reprocessStill();
    return;
  }
  if (state.mode === 'colors') {
    if (state.colorQuery?.id === 'pick') {
      state.pending.colorTap = { x: wx, y: wy };
      queueVision();
    }
    return;
  }
  state.pending.tap = { x: wx, y: wy };
  queueVision();
}

function syncTorch() {
  const ok = state.source === 'camera' && camera.torchSupported();
  $('btn-torch').hidden = !ok;
  if (!ok) state.torch = false;
  $('btn-torch').classList.toggle('active', state.torch);
}


function buildColorChips() {
  const row = $('color-chips');
  if (!row || row.dataset.ready) return;
  row.dataset.ready = '1';
  for (const c of COLOR_CHIPS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'color-chip';
    btn.dataset.id = c.id;
    btn.setAttribute('role', 'option');
    btn.setAttribute('aria-label', c.label);
    btn.innerHTML = `<span class="swatch" style="--swatch:${c.swatch}"></span><span class="chip-label">${c.id === 'pick' ? 'Pick' : c.label}</span>`;
    btn.addEventListener('click', () => selectColor(c.id));
    row.appendChild(btn);
  }
  syncColorChips();
}

function selectColor(id) {
  if (id === 'pick') state.colorQuery = { id: 'pick' };
  else state.colorQuery = { id };
  syncColorChips();
  updateUi();
  overlay.setTargets([], performance.now());
  queueVision();
}

function syncColorChips() {
  const row = $('color-chips');
  if (!row) return;
  const id = state.colorQuery?.id || 'pink';
  for (const btn of row.querySelectorAll('.color-chip')) {
    const on = btn.dataset.id === id;
    btn.classList.toggle('active', on);
    btn.setAttribute('aria-selected', String(on));
    if (btn.dataset.id === 'pick' && Number.isFinite(state.colorQuery?.L)) {
      const q = state.colorQuery;
      // Approximate sRGB swatch from the picked Lab so the chip shows what was tapped.
      btn.querySelector('.swatch').style.background = chipTint(q.name || 'pick');
      btn.querySelector('.chip-label').textContent = 'Picked';
    } else if (btn.dataset.id === 'pick') {
      btn.querySelector('.swatch').style.background = '';
      btn.querySelector('.chip-label').textContent = 'Pick';
    }
  }
}

function bind() {
  $('btn-camera').addEventListener('click', startCamera);
  $('btn-demo').addEventListener('click', loadDemo);
  $('btn-demo-gate').addEventListener('click', loadDemo);
  $('btn-upload').addEventListener('click', () => fileInput.click());
  $('btn-upload-gate').addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => { loadImageFile(fileInput.files && fileInput.files[0]); fileInput.value = ''; });

  shutter.addEventListener('click', () => {
    if (state.mode === 'scan') { doScan(); return; }
    if (state.source !== 'camera') { toast('Freeze is for the live camera'); return; }
    setFrozen(!state.frozen);
    if (state.frozen) toast('Frozen — outlines stay put while you grab socks');
  });
  $('mode-scan').addEventListener('click', () => setMode('scan'));
  $('mode-pairs').addEventListener('click', () => setMode('pairs'));
  $('mode-colors').addEventListener('click', () => setMode('colors'));
  buildColorChips();
  $('btn-clear-ref').addEventListener('click', (e) => { e.stopPropagation(); clearRef(); });

  $('btn-torch').addEventListener('click', async () => {
    state.torch = !state.torch;
    if (!(await camera.setTorch(state.torch))) { state.torch = false; toast('Flash is not available on this camera'); }
    syncTorch();
  });

  const saverEl = $('battery-saver');
  if (saverEl) {
    saverEl.checked = batterySaver;
    saverEl.addEventListener('change', () => setBatterySaver(saverEl.checked));
  }
  $('btn-settings').addEventListener('click', () => { $('sheet').hidden = false; });
  $('sheet-backdrop').addEventListener('click', () => { $('sheet').hidden = true; });
  $('btn-close-sheet').addEventListener('click', () => { $('sheet').hidden = true; });
  const sens = $('sens'), strict = $('strict');
  sens.addEventListener('input', () => {
    state.sensitivity = Number(sens.value) / 100;
    $('sens-val').textContent = sens.value;
    vision.reset(!!state.still, false);
    reprocessStill();
  });
  strict.addEventListener('input', () => {
    state.strictness = Number(strict.value) / 100;
    $('strict-val').textContent = strict.value;
    reprocessStill();
  });
  $('btn-lock-surface').addEventListener('click', () => {
    $('sheet').hidden = true;
    state.awaitingSurfaceTap = true;
    toast('Tap the sheet or floor — not a sock');
  });
  $('btn-reset-surface').addEventListener('click', () => {
    state.awaitingSurfaceTap = false;
    state.pending.surfaceAuto = true;
    $('surface-note').textContent = 'Surface color is estimated from the frame.';
    overlay.clearAll();
    toast('Back to auto surface');
    reprocessStill();
  });
  $('btn-switch-cam').addEventListener('click', async () => {
    try {
      if (state.source !== 'camera') await startCamera();
      else { await camera.flip(); video.classList.toggle('mirror', camera.isFront()); syncTorch(); resetVision(false); }
    } catch { toast('Could not switch camera'); }
  });

  overlayCanvas.addEventListener('pointerdown', onTap);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (state.source === 'camera' && state.torch) camera.setTorch(false);
      if (state.source === 'camera') {
        try { video.pause(); } catch { /* ignore */ }
      }
    } else {
      lastSig = null;
      if (state.source === 'camera' && !state.frozen) {
        video.play().catch(() => {});
      }
      ensureScanInEngine();
    }
  });
  window.addEventListener('resize', measure);
  window.addEventListener('orientationchange', () => setTimeout(measure, 250));
}

function nextResult() {
  return new Promise((resolve) => state.waiters.push(resolve));
}

function exposeDebug() {
  window.__sockMatcher = {
    getState() {
      const r = state.result;
      const lay = state.resultLayout;
      return {
        mode: state.mode,
        source: state.source,
        frozen: state.frozen,
        hasRef: state.hasRef,
        tracks: r ? r.tracks.map((t) => ({ id: t.id, cx: t.cx, cy: t.cy, area: t.area })) : [],
        pairs: r ? r.pairs.map((p) => ({ a: p.a, b: p.b, slot: p.slot, score: p.ema })) : [],
        matches: r ? r.matches : [],
        ref: r ? r.ref : null,
        work: lay ? { w: lay.w, h: lay.h } : null,
        rect: lay ? lay.rect : null,
        demoTruth: state.demoTruth,
        overlayItems: overlay.count(),
        timing: r ? r.timing : null,
        perf: perfSummary(),
        ml: { ...ml.getState(), result: state.mlResult },
        displayed: r ? displayLists(r) : null,
        savedScan: !!readSavedScan(),
        colorQuery: state.colorQuery,
        colorHits: r ? (r.colorHits || []).length : 0,
        batterySaver,
        power: {
          maxSide: MAX_VISION_SIDE,
          classicHz: 1000 / classicInterval(),
          mlInterval: mlInterval(),
          maxCrops: MAX_ML_CROPS,
        },
      };
    },
    loadDemo,
    setMode,
    nextResult,
    async tapWork(x, y) {
      state.pending.tap = { x, y };
      const p = nextResult();
      reprocessStill();
      await p;
      return this.getState();
    },
    async reprocess() { const p = nextResult(); reprocessStill(); await p; return this.getState(); },
    allPairs: (s) => vision.allPairs(s ?? state.strictness),
    perf: perfSummary,
  };
}

bind();
exposeDebug();
measure();
requestAnimationFrame(loop);
updateUi();
if (readSavedScan()) ensureScanInEngine();
if (new URLSearchParams(location.search).has('demo')) loadDemo();
