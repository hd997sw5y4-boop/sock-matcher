import { createEngine } from './engine.js';

const engine = createEngine();

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'frame') {
      const rgba = new Uint8ClampedArray(m.buf);
      const res = engine.process(rgba, m.w, m.h, m.opts);
      res.seq = m.seq;
      res.type = 'result';
      self.postMessage(res);
      // Hand the pixel buffer back so the main thread can reuse it.
      self.postMessage({ type: 'buffer', buf: m.buf }, [m.buf]);
    } else if (m.type === 'reset') {
      engine.reset(!!m.hard);
      if (m.clearRef) engine.clearRef();
    } else if (m.type === 'clearRef') {
      engine.clearRef();
    } else if (m.type === 'allPairs') {
      self.postMessage({ type: 'reply', id: m.id, value: engine.allPairs(m.strictness) });
    } else if (m.type === 'sim') {
      self.postMessage({ type: 'reply', id: m.id, value: engine.simBetween(m.a, m.b) });
    } else if (m.type === 'setRefMask') {
      const mask = m.mask ? new Uint8Array(m.mask) : null;
      const value = engine.setRefFromMask(mask, m.excludeId);
      self.postMessage({ type: 'reply', id: m.id, value });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.stack || err), seq: m.seq });
  }
};
