/**
 * Runs the vision engine in a Web Worker (falls back to the main thread).
 * Frames are grabbed into a small reusable canvas (about 90k pixels), so
 * the camera preview itself is never touched by the vision work.
 */
export function createVision(onResult) {
  let worker = null;
  let engine = null;
  let busy = false;
  let seq = 0;
  let replyId = 0;
  const replies = new Map();
  const pool = [];
  const grab = document.createElement('canvas');
  const gctx = grab.getContext('2d', { willReadFrequently: true });
  let mode = 'worker';

  function startWorker() {
    try {
      worker = new Worker(new URL('./cv/worker.js', import.meta.url), { type: 'module' });
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'result') { busy = false; onResult(m); }
        else if (m.type === 'buffer') pool.push(m.buf);
        else if (m.type === 'reply') { replies.get(m.id)?.(m.value); replies.delete(m.id); }
        else if (m.type === 'error') { busy = false; console.warn('vision worker:', m.message); }
      };
      worker.onerror = (err) => {
        console.warn('worker failed, running inline', err.message || err);
        worker = null; busy = false; mode = 'inline';
      };
    } catch {
      worker = null; mode = 'inline';
    }
  }
  startWorker();

  async function ensureInline() {
    if (!engine) {
      const mod = await import('./cv/engine.js');
      engine = mod.createEngine();
    }
    return engine;
  }

  /**
   * Draw `source` region (sx,sy,sw,sh) into a w x h buffer and process it.
   * Returns the frame sequence number, or 0 if a frame is already in flight.
   */
  function snapshot(source, rect, w, h) {
    if (!source || !rect || !w || !h) return null;
    if (grab.width !== w || grab.height !== h) { grab.width = w; grab.height = h; }
    gctx.drawImage(source, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, w, h);
    return gctx.getImageData(0, 0, w, h);
  }

  function submit(source, rect, w, h, opts) {
    if (busy) return 0;
    const img = snapshot(source, rect, w, h);
    if (!img) return 0;
    busy = true;
    const s = ++seq;
    if (worker) {
      let buf = pool.pop();
      if (!buf || buf.byteLength !== img.data.byteLength) buf = img.data.buffer.slice(0);
      else new Uint8ClampedArray(buf).set(img.data);
      worker.postMessage({ type: 'frame', buf, w, h, opts, seq: s }, [buf]);
    } else {
      ensureInline().then((eng) => {
        const res = eng.process(img.data, w, h, opts);
        res.seq = s;
        busy = false;
        onResult(res);
      });
    }
    return s;
  }

  function post(msg) {
    if (worker) worker.postMessage(msg);
    else ensureInline().then((eng) => {
      if (msg.type === 'reset') { eng.reset(!!msg.hard); if (msg.clearRef) eng.clearRef(); }
      if (msg.type === 'clearRef') eng.clearRef();
    });
  }

  function ask(msg, transfer) {
    if (!worker) {
      return ensureInline().then((eng) => {
        if (msg.type === 'allPairs') return eng.allPairs(msg.strictness);
        if (msg.type === 'setRefMask') return eng.setRefFromMask(msg.mask ? new Uint8Array(msg.mask) : null, msg.excludeId);
        if (msg.type === 'setAppearance') return eng.setAppearance(msg.desc, msg.excludeId);
        if (msg.type === 'getAppearance') return eng.getAppearance();
        return eng.simBetween(msg.a, msg.b);
      });
    }
    const id = ++replyId;
    return new Promise((resolve) => {
      replies.set(id, resolve);
      worker.postMessage({ ...msg, id }, transfer || []);
    });
  }

  return {
    submit,
    snapshot,
    reset: (hard = false, clearRef = false) => post({ type: 'reset', hard, clearRef }),
    clearRef: () => post({ type: 'clearRef' }),
    setRefMask: (mask, excludeId) => ask({ type: 'setRefMask', mask, excludeId }),
    setAppearance: (desc, excludeId) => ask({ type: 'setAppearance', desc, excludeId }),
    getAppearance: () => ask({ type: 'getAppearance' }),
    allPairs: (strictness) => ask({ type: 'allPairs', strictness }),
    sim: (a, b) => ask({ type: 'sim', a, b }),
    isBusy: () => busy,
    kind: () => (worker ? 'worker' : mode),
  };
}
