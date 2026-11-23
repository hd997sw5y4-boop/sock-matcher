/**
 * Pattern-model worker. Loads a same-origin quantized DINOv2 on WASM only
 * (never WebGPU), and embeds a few candidate crops on demand.
 */
import { pipeline, RawImage, env, LogLevel } from '/vendor/transformers.web.js';
import {
  configureEnv, loadExtractor, pickSock, scoreTracks, embedCrop, maskedCrop,
  floodFromPoint,
} from './runtime.js';

const tf = { pipeline, RawImage, env };
configureEnv(env);
try { env.logLevel = LogLevel.ERROR; } catch { /* older typings */ }

let extractor = null;
let readyDino = false;
let refEmb = null;
const cache = new Map();
let busy = false;

function post(msg) { self.postMessage(msg); }

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load();
    else if (m.type === 'pick') await pick(m);
    else if (m.type === 'embedRef') await embedRef(m);
    else if (m.type === 'setRef') {
      refEmb = m.embedding ? new Float32Array(m.embedding) : refEmb;
      cache.clear();
      post({ type: 'reply', id: m.id, value: { ok: !!refEmb } });
    } else if (m.type === 'clearRef') {
      refEmb = null;
      cache.clear();
    } else if (m.type === 'score') await score(m);
    else if (m.type === 'reset') { refEmb = null; cache.clear(); }
  } catch (err) {
    post({ type: 'error', id: m.id, message: String(err && err.stack || err) });
  }
};

async function load() {
  if (readyDino) {
    post({ type: 'ready', dino: true, sam: false });
    return;
  }
  post({ type: 'progress', phase: 'dino', progress: 1, status: 'start' });
  extractor = await loadExtractor(tf, { progress: (p) => post({ type: 'progress', ...p }) });
  readyDino = true;
  post({ type: 'progress', phase: 'done', progress: 100, status: 'ready' });
  post({ type: 'ready', dino: true, sam: false });
}

async function pick(m) {
  const rgba = new Uint8ClampedArray(m.buf);
  const point = m.point;
  // Flood-fill only — SlimSAM stays off the scan path and is not loaded.
  const result = await pickSock(tf, { extractor, sam: null }, rgba, m.w, m.h, point, m.box);
  if (result.ok && result.embedding) refEmb = result.embedding;
  const value = {
    ok: result.ok,
    method: result.method,
    box: result.box,
    area: result.area,
    cx: result.cx,
    cy: result.cy,
    embedding: result.embedding ? Array.from(result.embedding) : null,
    mask: result.mask ? result.mask : null,
  };
  post({ type: 'reply', id: m.id, value });
  if (m.buf) post({ type: 'buffer', buf: m.buf }, [m.buf]);
}

async function embedRef(m) {
  if (!extractor) {
    post({ type: 'reply', id: m.id, value: { ok: false } });
    return;
  }
  const rgba = new Uint8ClampedArray(m.buf);
  let crop;
  if (m.mask && m.box) {
    const region = {
      ok: true,
      mask: new Uint8Array(m.mask),
      box: m.box,
      cx: m.cx,
      cy: m.cy,
    };
    crop = maskedCrop(rgba, m.w, m.h, region, 8);
  } else if (m.box) {
    const seed = { x: (m.box.x0 + m.box.x1) / 2, y: (m.box.y0 + m.box.y1) / 2 };
    const region = floodFromPoint(rgba, m.w, m.h, seed.x, seed.y);
    crop = region.ok ? maskedCrop(rgba, m.w, m.h, region, 8) : null;
  }
  if (!crop) {
    post({ type: 'reply', id: m.id, value: { ok: false } });
    return;
  }
  const embedding = await embedCrop(tf, extractor, crop.rgba, crop.w, crop.h);
  if (embedding) refEmb = embedding;
  cache.clear();
  post({
    type: 'reply',
    id: m.id,
    value: { ok: !!embedding, embedding: embedding ? Array.from(embedding) : null },
  });
  if (m.buf) post({ type: 'buffer', buf: m.buf }, [m.buf]);
}

async function score(m) {
  if (busy) { post({ type: 'reply', id: m.id, value: { skipped: true } }); return; }
  if (!extractor || !refEmb && m.mode !== 'pairs') {
    post({ type: 'reply', id: m.id, value: { matches: [], pairs: [], ready: false } });
    return;
  }
  busy = true;
  try {
    const rgba = new Uint8ClampedArray(m.buf);
    const out = await scoreTracks(tf, extractor, rgba, m.w, m.h, m.tracks || [], refEmb, cache, {
      mode: m.mode,
      maxCrops: m.maxCrops,
    });
    if (cache.size > 48) {
      const keys = [...cache.keys()];
      for (let i = 0; i < keys.length - 32; i++) cache.delete(keys[i]);
    }
    post({ type: 'reply', id: m.id, value: { ...out, ready: true } });
    if (m.buf) post({ type: 'buffer', buf: m.buf }, [m.buf]);
  } finally {
    busy = false;
  }
}
