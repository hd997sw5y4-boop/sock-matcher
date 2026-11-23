import { createProcessor } from './cv/process.js';
import { createTracker } from './cv/track.js';
import { descriptorSimilarity } from './cv/match.js';
import { createCamera } from './camera.js';
import { generateDemoPile, sockBounds } from './demo.js';
import { createMapper, drawDisplayFrame, renderOverlay, hitTestSock } from './overlay.js';

const $ = (id) => document.getElementById(id);

const video = $('video');
const frame = $('frame');
const overlay = $('overlay');
const gate = $('gate');
const statusChip = $('status-chip');
const toastEl = $('toast');
const fileInput = $('file-input');
const btnFreeze = $('btn-freeze');

const camera = createCamera(video);
const processor = createProcessor();
const tracker = createTracker();

const state = {
  mode: 'pairs', // 'pairs' | 'tap'
  source: null, // 'camera' | 'photo' | 'demo'
  frozen: false,
  torch: false,
  sensitivity: 0.5,
  strictness: 0.55,
  awaitingSurfaceTap: false,
  srcW: 0,
  srcH: 0,
  workW: 0,
  workH: 0,
  socks: [],
  pairs: [],
  unmatched: [],
  selected: null,
  matches: [],
  demoTruth: null,
  lastResult: null,
  lastProcess: 0,
  stillImage: null, // HTMLCanvasElement | HTMLImageElement
};

let toastTimer = 0;
let anim = 0;

function showGate(show) {
  gate.classList.toggle('hidden', !show);
}

function toast(msg, ms = 2200) {
  toastEl.hidden = false;
  toastEl.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.hidden = true; }, ms);
}

function setChip() {
  const nPairs = state.pairs.length;
  const nSolo = state.unmatched.length;
  let text = '';
  if (!state.socks.length) {
    text = 'No socks found — spread them out';
  } else if (state.mode === 'tap') {
    text = state.selected
      ? (state.matches.length ? `Best match ${Math.round(state.matches[0].score * 100)}%` : 'Tap looks unmatched')
      : 'Tap a sock to find its match';
  } else {
    text = nPairs
      ? `${nPairs} pair${nPairs === 1 ? '' : 's'}${nSolo ? ` · ${nSolo} leftover` : ''}`
      : `${state.socks.length} sock${state.socks.length === 1 ? '' : 's'} · no pairs yet`;
  }
  if (state.frozen) text = `Frozen · ${text}`;
  statusChip.hidden = false;
  statusChip.textContent = text;
  statusChip.classList.toggle('frozen', state.frozen);
}

function hideChipIfGated() {
  if (!gate.classList.contains('hidden')) statusChip.hidden = true;
}

function resizeCanvases() {
  const stage = $('stage');
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  const dpr = Math.min(2.5, window.devicePixelRatio || 1);
  for (const c of [frame, overlay]) {
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
      c.style.width = `${w}px`;
      c.style.height = `${h}px`;
    }
  }
  return { w, h, dpr };
}

function currentSource() {
  if (state.stillImage) return state.stillImage;
  if (state.source === 'camera' && video.videoWidth) return video;
  return null;
}

function sourceSize(src) {
  if (src === video) return { w: video.videoWidth, h: video.videoHeight };
  return { w: src.width, h: src.height };
}

function processNow() {
  const src = currentSource();
  if (!src) return;
  const { w: srcW, h: srcH } = sourceSize(src);
  if (!srcW || !srcH) return;
  state.srcW = srcW;
  state.srcH = srcH;
  const result = processor.run(src, {
    srcW, srcH,
    sensitivity: state.sensitivity,
    strictness: state.strictness,
  });
  state.lastResult = result;
  state.workW = result.w;
  state.workH = result.h;
  const tracked = tracker.update(result.blobs, result.descs, result.pairing, result.w, result.h);
  state.socks = tracked.socks;
  state.pairs = tracked.pairs;
  state.unmatched = tracked.unmatched;
  refreshSelection();
  setChip();
}

function refreshSelection() {
  if (state.mode !== 'tap' || !state.selected) {
    state.matches = [];
    return;
  }
  // Re-attach selected sock by trackId, then nearest center.
  let sel = state.socks.find((s) => s.trackId === state.selected.trackId && s.trackId > 0);
  if (!sel) {
    sel = state.socks.find((s) => Math.hypot(s.cx - state.selected.cx, s.cy - state.selected.cy) < 24);
  }
  if (!sel) {
    state.selected = null;
    state.matches = [];
    return;
  }
  state.selected = sel;
  const scored = state.socks
    .filter((s) => s.index !== sel.index)
    .map((s) => ({ index: s.index, score: descriptorSimilarity(sel.desc, s.desc), sock: s }))
    .sort((a, b) => b.score - a.score);
  const threshold = 0.42 + state.strictness * 0.36;
  const top = scored[0];
  // Show the best match if it clears the bar; if a close second exists, show it too.
  state.matches = [];
  if (top && top.score >= threshold * 0.92) {
    state.matches.push(top);
    if (scored[1] && scored[1].score >= top.score - 0.04 && scored[1].score >= threshold) {
      state.matches.push(scored[1]);
    }
  }
}

function paint(ts) {
  anim = requestAnimationFrame(paint);
  const src = currentSource();
  const { w, h, dpr } = resizeCanvases();
  const fctx = frame.getContext('2d');
  fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!src) {
    fctx.fillStyle = '#111';
    fctx.fillRect(0, 0, w, h);
    return;
  }
  const { w: srcW, h: srcH } = sourceSize(src);
  const mirror = state.source === 'camera' && camera.isFront();
  const fit = state.source === 'camera' ? 'cover' : 'contain';
  drawDisplayFrame(fctx, src, w, h, srcW, srcH, mirror, fit);

  const live = state.source === 'camera' && !state.frozen;
  if (srcW && srcH && (live || !state.lastResult)) {
    if (!live || ts - state.lastProcess > 70) {
      processNow();
      state.lastProcess = ts;
    }
  }

  const octx = overlay.getContext('2d');
  const mapper = createMapper(state.srcW || srcW, state.srcH || srcH, w, h, fit);
  // If the display is mirrored (front camera), wrap mapper.
  const viewMapper = mirror ? mirrorMapper(mapper, w) : mapper;
  renderOverlay(octx, {
    viewW: w,
    viewH: h,
    dpr,
    mapper: viewMapper,
    workW: state.workW,
    workH: state.workH,
    mode: state.mode,
    socks: state.socks,
    pairs: state.pairs,
    unmatched: state.unmatched,
    selected: state.selected,
    matches: state.matches,
    frozen: state.frozen,
  }, ts);
}

function mirrorMapper(mapper, viewW) {
  return {
    ...mapper,
    workToView(x, y, ww, wh) {
      const [vx, vy] = mapper.workToView(x, y, ww, wh);
      return [viewW - vx, vy];
    },
    viewToWork(x, y, ww, wh) {
      return mapper.viewToWork(viewW - x, y, ww, wh);
    },
  };
}

function openLive() {
  showGate(false);
  hideChipIfGated();
  setChip();
}

async function startCamera() {
  try {
    state.stillImage = null;
    state.source = 'camera';
    state.frozen = false;
    state.selected = null;
    tracker.reset();
    processor.unlockSurface();
    updateSurfaceNote();
    await camera.start('environment');
    syncTorchButton();
    setFrozen(false);
    openLive();
    toast('Spread socks out on a plain surface');
  } catch (err) {
    console.warn(err);
    toast('Camera blocked — try Demo or Upload a photo');
  }
}

function loadDemo() {
  camera.stop();
  syncTorchButton();
  const demo = generateDemoPile();
  state.stillImage = demo.canvas;
  state.source = 'demo';
  state.demoTruth = demo.socks;
  state.frozen = true;
  state.selected = null;
  tracker.reset();
  processor.unlockSurface();
  updateSurfaceNote();
  setFrozen(true);
  openLive();
  processNow();
  toast('Demo pile ready — 8 pairs');
}

function loadImageFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas');
    const max = 1600;
    const scale = Math.min(1, max / Math.max(img.width, img.height));
    c.width = Math.round(img.width * scale);
    c.height = Math.round(img.height * scale);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(url);
    camera.stop();
    syncTorchButton();
    state.stillImage = c;
    state.source = 'photo';
    state.demoTruth = null;
    state.frozen = true;
    state.selected = null;
    tracker.reset();
    processor.unlockSurface();
    updateSurfaceNote();
    setFrozen(true);
    openLive();
    processNow();
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    toast('Could not read that photo');
  };
  img.src = url;
}

function setFrozen(on) {
  state.frozen = on;
  btnFreeze.classList.toggle('on', on);
  btnFreeze.querySelector('.shutter-label').textContent = on ? 'Resume' : 'Freeze';
  setChip();
}

function setMode(mode) {
  state.mode = mode;
  $('mode-pairs').classList.toggle('active', mode === 'pairs');
  $('mode-tap').classList.toggle('active', mode === 'tap');
  $('mode-pairs').setAttribute('aria-selected', String(mode === 'pairs'));
  $('mode-tap').setAttribute('aria-selected', String(mode === 'tap'));
  if (mode !== 'tap') {
    state.selected = null;
    state.matches = [];
  }
  setChip();
}

function syncTorchButton() {
  const btn = $('btn-torch');
  const ok = state.source === 'camera' && camera.torchSupported();
  btn.hidden = !ok;
  if (!ok) state.torch = false;
  btn.classList.toggle('active', state.torch);
}

function updateSurfaceNote() {
  const note = $('surface-note');
  note.textContent = processor.isLocked()
    ? 'Surface color is locked. Tap Auto Surface to go back to estimating from the frame edges.'
    : 'Surface color is estimated from the edges of the frame.';
}

function onStageTap(ev) {
  if (!gate.classList.contains('hidden')) return;
  const rect = overlay.getBoundingClientRect();
  const x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
  const y = (ev.touches ? ev.touches[0].clientY : ev.clientY) - rect.top;
  const { w, h } = { w: rect.width, h: rect.height };
  const fit = state.source === 'camera' ? 'cover' : 'contain';
  const mapper = createMapper(state.srcW, state.srcH, w, h, fit);
  const viewMapper = (state.source === 'camera' && camera.isFront()) ? mirrorMapper(mapper, w) : mapper;
  if (!state.workW) return;
  const [wx, wy] = viewMapper.viewToWork(x, y, state.workW, state.workH);

  if (state.awaitingSurfaceTap) {
    processor.sampleSurface(wx, wy);
    state.awaitingSurfaceTap = false;
    updateSurfaceNote();
    tracker.reset();
    processNow();
    toast('Surface locked');
    return;
  }

  if (state.mode !== 'tap') return;
  const hit = hitTestSock(state.socks, wx, wy);
  if (!hit) {
    state.selected = null;
    state.matches = [];
    setChip();
    return;
  }
  state.selected = hit;
  refreshSelection();
  setChip();
}

function bind() {
  $('btn-camera').addEventListener('click', startCamera);
  $('btn-demo').addEventListener('click', loadDemo);
  $('btn-demo-gate').addEventListener('click', loadDemo);
  $('btn-upload').addEventListener('click', () => fileInput.click());
  $('btn-upload-gate').addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const f = fileInput.files && fileInput.files[0];
    loadImageFile(f);
    fileInput.value = '';
  });

  btnFreeze.addEventListener('click', () => {
    if (state.source !== 'camera') {
      toast('Freeze is for the live camera');
      return;
    }
    setFrozen(!state.frozen);
    if (state.frozen) toast('Frozen — boxes stay put while you pick socks up');
  });

  $('mode-pairs').addEventListener('click', () => setMode('pairs'));
  $('mode-tap').addEventListener('click', () => setMode('tap'));

  $('btn-torch').addEventListener('click', async () => {
    state.torch = !state.torch;
    const ok = await camera.setTorch(state.torch);
    if (!ok) {
      state.torch = false;
      toast('Flash is not available on this camera');
    }
    syncTorchButton();
  });

  $('btn-settings').addEventListener('click', () => { $('sheet').hidden = false; });
  $('sheet-backdrop').addEventListener('click', () => { $('sheet').hidden = true; });
  $('btn-close-sheet').addEventListener('click', () => { $('sheet').hidden = true; });

  const sens = $('sens'), strict = $('strict');
  const syncSliders = () => {
    $('sens-val').textContent = String(Math.round(state.sensitivity * 100));
    $('strict-val').textContent = String(Math.round(state.strictness * 100));
  };
  sens.addEventListener('input', () => {
    state.sensitivity = Number(sens.value) / 100;
    syncSliders();
    tracker.reset();
    processNow();
  });
  strict.addEventListener('input', () => {
    state.strictness = Number(strict.value) / 100;
    syncSliders();
    processNow();
  });

  $('btn-lock-surface').addEventListener('click', () => {
    $('sheet').hidden = true;
    state.awaitingSurfaceTap = true;
    toast('Tap the sheet or floor — not a sock');
  });
  $('btn-reset-surface').addEventListener('click', () => {
    processor.unlockSurface();
    state.awaitingSurfaceTap = false;
    updateSurfaceNote();
    tracker.reset();
    processNow();
    toast('Back to auto surface');
  });
  $('btn-switch-cam').addEventListener('click', async () => {
    try {
      if (state.source !== 'camera') await startCamera();
      else {
        await camera.flip();
        syncTorchButton();
        tracker.reset();
      }
    } catch {
      toast('Could not switch camera');
    }
  });

  overlay.addEventListener('pointerdown', onStageTap);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.source === 'camera' && state.torch) {
      camera.setTorch(false);
    }
  });
  window.addEventListener('resize', resizeCanvases);
  window.addEventListener('orientationchange', () => setTimeout(resizeCanvases, 250));
}

function exposeDebug() {
  window.__sockMatcher = {
    getState() {
      return {
        mode: state.mode,
        source: state.source,
        frozen: state.frozen,
        socks: state.socks.map((s) => ({
          index: s.index, trackId: s.trackId, cx: s.cx, cy: s.cy,
          minX: s.minX, minY: s.minY, maxX: s.maxX, maxY: s.maxY, area: s.area,
        })),
        pairs: state.pairs.map((p) => ({ a: p.a, b: p.b, num: p.num, score: p.score })),
        unmatched: state.unmatched.map((s) => s.index),
        selected: state.selected ? state.selected.index : null,
        matches: state.matches.map((m) => ({ index: m.index, score: m.score })),
        work: { w: state.workW, h: state.workH },
        src: { w: state.srcW, h: state.srcH },
        demoTruth: state.demoTruth,
      };
    },
    loadDemo,
    setMode,
    tapWork(wx, wy) {
      const hit = hitTestSock(state.socks, wx, wy);
      if (hit) {
        state.selected = hit;
        refreshSelection();
        setChip();
      }
      return this.getState();
    },
    debugMask() {
      const r = state.lastResult;
      if (!r) return null;
      return { w: r.w, h: r.h, threshold: r.threshold, bg: r.bg, blobCount: r.blobs.length, mask: Array.from(r.mask) };
    },
    setSliders(sens, strict) {
      state.sensitivity = sens;
      state.strictness = strict;
      $('sens').value = String(Math.round(sens * 100));
      $('strict').value = String(Math.round(strict * 100));
      $('sens-val').textContent = String(Math.round(sens * 100));
      $('strict-val').textContent = String(Math.round(strict * 100));
      tracker.reset();
      processNow();
    },
  };
}

bind();
exposeDebug();
requestAnimationFrame(paint);

if (new URLSearchParams(location.search).has('demo')) {
  loadDemo();
}
