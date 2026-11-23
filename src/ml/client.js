/**
 * Main-thread handle for the pattern-model worker.
 * Lazy-loads on first use, reports download progress, and scores a few crops on demand.
 */
import { createScoreTracker, pickMatches, MAX_SHOWN } from './rank.js';

export function createMlClient() {
  let worker = null;
  let replyId = 0;
  const replies = new Map();
  const pool = [];
  const state = {
    status: 'idle', // idle | loading | ready | error
    dino: false,
    sam: false,
    progress: 0,
    phase: '',
    error: '',
  };
  const listeners = new Set();
  const tracker = createScoreTracker();
  let lastScoreAt = 0;
  let scoring = false;

  function emit() { for (const fn of listeners) fn(state); }

  function start() {
    if (worker || state.status === 'loading' || state.status === 'ready') return;
    try {
      worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    } catch (err) {
      state.status = 'error';
      state.error = String(err && err.message || err);
      emit();
      return;
    }
    state.status = 'loading';
    emit();
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') {
        state.progress = m.progress ?? state.progress;
        state.phase = m.phase || state.phase;
        emit();
      } else if (m.type === 'ready') {
        state.dino = !!m.dino;
        state.sam = !!m.sam;
        if (state.dino) {
          state.status = 'ready';
          state.progress = m.sam ? 100 : Math.max(state.progress, 62);
        }
        emit();
      } else if (m.type === 'reply') {
        replies.get(m.id)?.(m.value);
        replies.delete(m.id);
      } else if (m.type === 'buffer') {
        pool.push(m.buf);
      } else if (m.type === 'error') {
        if (m.id && replies.has(m.id)) { replies.get(m.id)(null); replies.delete(m.id); }
        else {
          state.status = state.dino ? 'ready' : 'error';
          state.error = m.message || 'model error';
          emit();
        }
      }
    };
    worker.onerror = (err) => {
      console.warn('pattern worker', err.message || err);
      if (!state.dino) {
        state.status = 'error';
        state.error = String(err.message || err);
        emit();
      }
    };
    worker.postMessage({ type: 'load' });
  }

  function ask(msg, transfer) {
    if (!worker) return Promise.resolve(null);
    const id = ++replyId;
    return new Promise((resolve) => {
      replies.set(id, resolve);
      worker.postMessage({ ...msg, id }, transfer || []);
    });
  }

  function copyFrame(rgba) {
    const src = rgba.buffer ? rgba : new Uint8ClampedArray(rgba);
    let buf = pool.pop();
    if (!buf || buf.byteLength !== src.byteLength) buf = src.buffer.slice(0);
    else new Uint8ClampedArray(buf).set(src);
    return buf;
  }

  return {
    start,
    onChange: (fn) => { listeners.add(fn); return () => listeners.delete(fn); },
    getState: () => state,
    isReady: () => state.status === 'ready' && state.dino,
    hasSam: () => state.sam,
    reset() { tracker.reset(); if (worker) worker.postMessage({ type: 'reset' }); },
    clearRef() { tracker.reset(); if (worker) worker.postMessage({ type: 'clearRef' }); },
    async setRefEmbedding(embedding) {
      if (!worker || !embedding) return;
      await ask({ type: 'setRef', embedding: Array.from(embedding) });
    },
    async pick(rgba, w, h, point, box) {
      if (!worker) return { ok: false };
      const buf = copyFrame(rgba);
      const value = await ask({ type: 'pick', buf, w, h, point, box }, [buf]);
      tracker.reset();
      return value || { ok: false };
    },
    async embedRef(rgba, w, h, { mask, box, cx, cy } = {}) {
      if (!worker) return { ok: false };
      const buf = copyFrame(rgba);
      const value = await ask({ type: 'embedRef', buf, w, h, mask, box, cx, cy }, [buf]);
      tracker.reset();
      return value || { ok: false };
    },
    /**
     * Re-score at most ~1–2 Hz, and only a few candidate crops — never the
     * full frame continuously. Returns null if this tick was skipped.
     */
    async score(rgba, w, h, tracks, { mode = 'scan', minInterval = 700, maxCrops = 6, now = performance.now(), pairColor } = {}) {
      if (!this.isReady() || scoring) return null;
      if (now - lastScoreAt < minInterval) return null;
      lastScoreAt = now;
      scoring = true;
      try {
        const buf = copyFrame(rgba);
        const value = await ask({ type: 'score', buf, w, h, tracks, mode, maxCrops }, [buf]);
        if (!value || value.skipped) return null;
        if (mode === 'pairs') {
          let pairs = value.pairs || [];
          if (pairColor) {
            for (const p of pairs) {
              const c = pairColor(p.a, p.b);
              if (c != null) { p.color = c; p.score = p.score * 0.7 / 0.7; } // color already in combined if provided earlier
            }
          }
          const shown = tracker.update(pairs, now);
          return { mode, pairs: shown.slice(0, MAX_SHOWN), matches: [] };
        }
        const shown = tracker.update(value.matches || [], now);
        return { mode, matches: shown.slice(0, MAX_SHOWN), pairs: [] };
      } finally {
        scoring = false;
      }
    },
    pickMatches,
  };
}
