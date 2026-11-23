/**
 * Pattern-model worker. Loads DINOv2 + SlimSAM once, caches weights with
 * the Cache API, and only re-embeds when a track's appearance changes.
 */
import { pipeline, SamModel, AutoProcessor, RawImage, env, LogLevel } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';
import {
  configureEnv, loadExtractor, loadSam, pickSock, scoreTracks,
} from './runtime.js';

const tf = { pipeline, SamModel, AutoProcessor, RawImage, env };
configureEnv(env);
try { env.logLevel = LogLevel.ERROR; } catch { /* older typings */ }

let extractor = null;
let sam = null;
let readyDino = false;
let readySam = false;
let refEmb = null;
const cache = new Map();
let busy = false;

function post(msg) { self.postMessage(msg); }

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load();
    else if (m.type === 'pick') await pick(m);
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
  if (readyDino && (readySam || sam === null)) {
    post({ type: 'ready', dino: readyDino, sam: readySam });
    return;
  }
  post({ type: 'progress', phase: 'dino', progress: 1, status: 'start' });
  extractor = await loadExtractor(tf, { progress: (p) => post({ type: 'progress', ...p }) });
  readyDino = true;
  post({ type: 'progress', phase: 'dino', progress: 62, status: 'ready' });
  post({ type: 'ready', dino: true, sam: false });
  try {
    sam = await loadSam(tf, { progress: (p) => post({ type: 'progress', ...p }) });
    readySam = !!(sam && sam.model);
  } catch (err) {
    console.warn(err);
    sam = null;
    readySam = false;
  }
  post({ type: 'progress', phase: 'done', progress: 100, status: 'ready' });
  post({ type: 'ready', dino: true, sam: readySam });
}

async function pick(m) {
  const rgba = new Uint8ClampedArray(m.buf);
  const point = m.point;
  const result = await pickSock(tf, { extractor, sam }, rgba, m.w, m.h, point, m.box);
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

async function score(m) {
  if (busy) { post({ type: 'reply', id: m.id, value: { skipped: true } }); return; }
  if (!extractor || !refEmb && m.mode !== 'pairs') {
    post({ type: 'reply', id: m.id, value: { matches: [], pairs: [], ready: false } });
    return;
  }
  busy = true;
  try {
    const rgba = new Uint8ClampedArray(m.buf);
    const out = await scoreTracks(tf, extractor, rgba, m.w, m.h, m.tracks || [], refEmb, cache, { mode: m.mode });
    // Trim the embedding cache so a long session doesn't grow forever.
    if (cache.size > 80) {
      const keys = [...cache.keys()];
      for (let i = 0; i < keys.length - 48; i++) cache.delete(keys[i]);
    }
    post({ type: 'reply', id: m.id, value: { ...out, ready: true } });
    if (m.buf) post({ type: 'buffer', buf: m.buf }, [m.buf]);
  } finally {
    busy = false;
  }
}
