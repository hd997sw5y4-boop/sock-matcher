/**
 * Vision engine: segmentation -> tracking (with cached descriptors) ->
 * matching with temporal confidence. Pure JS; runs in a Worker or inline.
 */
import { rgbaToLabPlanes, blur3 } from './lab.js';
import { createSegmenter } from './segment.js';
import { describeRegion } from './describe.js';
import { similarity, appearanceSimilarity, appearanceOf, thresholdFor, confidentPairs } from './match.js';
import { collectColorHits, pickColorAt } from './colorClass.js';

export const MAX_SHOWN = 3;
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
// Light hysteresis only: brief misses shouldn't flicker, but a wrong lock must not stick.
const TRACK_HOLD_MS = 700;
const TRACK_HOLD_MISSES = 10;
const MATCH_HOLD_MS = 400;
const MATCH_HOLD_MISSES = 5;
const DROP_SLACK = 0.08; // acquire at thr, drop a little below it
const MISS_EMA_DECAY = 0.90;

export function createEngine() {
  const seg = createSegmenter();
  let W = 0, H = 0;
  let planes = null, tmpF = null;
  let tracks = new Map();
  let nextId = 1;
  let frameNo = 0;
  let lastT = 0;
  let gvx = 0, gvy = 0; // global motion, work px per ms
  const simCache = new Map();
  let pairState = new Map(); // key -> {a,b,ema,shown,slot,seen}
  let matchState = new Map(); // trackId -> {ema, shown, slot}
  let ref = null; // {desc, excludeId, version}
  let refVersion = 0;
  let surface = null;
  let lastSeg = null;

  function ensure(w, h) {
    if (w === W && h === H && planes) return;
    W = w; H = h;
    const n = w * h;
    planes = {
      L: new Float32Array(n), A: new Float32Array(n), B: new Float32Array(n),
      Ls: new Float32Array(n), As: new Float32Array(n), Bs: new Float32Array(n),
    };
    tmpF = new Float32Array(n);
  }

  function reset(hard = false) {
    tracks = new Map(); simCache.clear(); pairState = new Map(); matchState = new Map();
    gvx = gvy = 0; lastT = 0;
    if (hard) nextId = 1;
  }

  function clearRef() { ref = null; matchState = new Map(); }

  function sim(ta, tb) {
    const k = ta.id < tb.id ? `${ta.id}:${tb.id}` : `${tb.id}:${ta.id}`;
    const c = simCache.get(k);
    const va = ta.id < tb.id ? ta.dv : tb.dv, vb = ta.id < tb.id ? tb.dv : ta.dv;
    if (c && c.va === va && c.vb === vb) return c.s;
    const s = similarity(ta.desc, tb.desc);
    simCache.set(k, { s, va, vb });
    return s;
  }

  function refSim(t) {
    if (t.refV === refVersion && t.refDv === t.dv) return t.refS;
    t.refS = appearanceSimilarity(ref.desc, t.desc);
    t.refV = refVersion; t.refDv = t.dv;
    return t.refS;
  }

  function iou(a, b) {
    const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
    const x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
    const iw = Math.max(0, x1 - x0), ih = Math.max(0, y1 - y0);
    const inter = iw * ih;
    if (!inter) return 0;
    const uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return uni > 0 ? inter / uni : 0;
  }

  function isSticky(id) {
    const ms = matchState.get(id);
    if (ms && ms.shown) return true;
    for (const st of pairState.values()) {
      if (st.shown && (st.a === id || st.b === id)) return true;
    }
    return false;
  }

  function holdExpired(st) {
    return (st.missed || 0) >= MATCH_HOLD_MISSES && (lastT - (st.lastHit || 0) >= MATCH_HOLD_MS);
  }

  function associate(regions, dt) {
    const diag = Math.hypot(W, H);
    const cands = [];
    for (const r of regions) {
      const rbox = [r.minX, r.minY, r.maxX, r.maxY];
      for (const t of tracks.values()) {
        const vx = t.vx || 0, vy = t.vy || 0;
        const px = t.cx + vx * dt, py = t.cy + vy * dt;
        const pbox = t.box
          ? [t.box[0] + vx * dt, t.box[1] + vy * dt, t.box[2] + vx * dt, t.box[3] + vy * dt]
          : null;
        const ov = pbox ? iou(rbox, pbox) : 0;
        const d = Math.hypot(r.cx - px, r.cy - py);
        const established = (t.hits || 0) >= 3 || isSticky(t.id);
        const gate = diag * (established ? 0.14 : 0.09) + Math.hypot(vx, vy) * dt * 0.8 + (established ? 12 : 0);
        if (d > gate && ov < 0.15) continue;
        const ar = Math.abs(Math.log(r.area / Math.max(1, t.area)));
        if (ar > 1.2) continue;
        const dc = Math.hypot((r.mL - t.mL) * 0.5, r.mA - t.mA, r.mB - t.mB);
        const colorGate = established ? 30 : 22;
        if (dc > colorGate && ov < 0.25) continue;
        let cost = (1 - ov) * 0.9 + d / Math.max(gate, 1) * 0.7 + dc / 30 + ar * 0.5;
        if (established) cost -= 0.08;
        if (isSticky(t.id)) cost -= 0.12;
        cands.push({ r, t, cost, dx: r.cx - t.cx, dy: r.cy - t.cy });
      }
    }
    cands.sort((a, b) => a.cost - b.cost);
    const usedR = new Set(), usedT = new Set();
    const pairs = [];
    for (const c of cands) {
      if (usedR.has(c.r) || usedT.has(c.t)) continue;
      usedR.add(c.r); usedT.add(c.t);
      pairs.push(c);
    }
    return pairs;
  }

  function quickMean(r) {
    const { Ls, As, Bs } = planes;
    let l = 0, a = 0, b = 0, c = 0;
    const px = r.pixels;
    const step = Math.max(1, (px.length / 64) | 0);
    for (let k = 0; k < px.length; k += step) { const i = px[k]; l += Ls[i]; a += As[i]; b += Bs[i]; c++; }
    r.mL = l / c; r.mA = a / c; r.mB = b / c;
  }

  /**
   * @param {Uint8ClampedArray} rgba
   * @param {object} o {mode:'scan'|'pairs', sensitivity, strictness, t, still, scanBox, tap, surfaceTap}
   */
  function process(rgba, w, h, o = {}) {
    const t0 = now();
    ensure(w, h);
    const n = w * h;
    rgbaToLabPlanes(rgba, n, planes.L, planes.A, planes.B);
    blur3(planes.L, planes.Ls, tmpF, w, h);
    blur3(planes.A, planes.As, tmpF, w, h);
    blur3(planes.B, planes.Bs, tmpF, w, h);
    if (o.surfaceTap) {
      const i = Math.max(0, Math.min(h - 1, Math.round(o.surfaceTap.y))) * w + Math.max(0, Math.min(w - 1, Math.round(o.surfaceTap.x)));
      surface = { L: planes.Ls[i], a: planes.As[i], b: planes.Bs[i] };
      reset();
    }
    if (o.surfaceAuto) { surface = null; reset(); }
    const s = seg.segment(planes, w, h, { sensitivity: o.sensitivity ?? 0.5, surface });
    lastSeg = s;
    const t1 = now();
    if ((o.mode || 'scan') === 'colors') {
      frameNo++;
      const colorHits = collectColorHits(planes, s, o.colorQuery || { id: 'pink' }, { roi: o.roi });
      const colorPick = o.colorTap ? pickColorAt(o.colorTap.x, o.colorTap.y, planes, s, w, h) : null;
      return {
        w, h, frame: frameNo,
        tracks: [], pairs: [], matches: [], scan: null,
        ref: ref ? { excludeId: ref.excludeId || 0 } : null,
        colorHits, colorPick,
        regions: s.regions.length,
        timing: { seg: t1 - t0, track: 0, match: 0, total: now() - t0, described: 0 },
      };
    }
    for (const r of s.regions) quickMean(r);

    const t = o.t ?? now();
    const dt = lastT ? Math.max(1, Math.min(400, t - lastT)) : 0;
    if (o.still) reset(true);
    frameNo++;
    const matched = o.still ? [] : associate(s.regions, dt);
    // Global motion = median displacement of matched tracks.
    if (matched.length >= 3 && dt) {
      const xs = matched.map((m) => m.dx).sort((a, b) => a - b);
      const ys = matched.map((m) => m.dy).sort((a, b) => a - b);
      const mx = xs[xs.length >> 1] / dt, my = ys[ys.length >> 1] / dt;
      gvx = gvx * 0.4 + mx * 0.6; gvy = gvy * 0.4 + my * 0.6;
    } else { gvx *= 0.5; gvy *= 0.5; }
    lastT = t;

    const refL = s.surface.L;
    let described = 0;
    const seen = new Set();
    const byRegion = new Map();
    for (const m of matched) byRegion.set(m.r, m.t);
    for (const r of s.regions) {
      let tr = byRegion.get(r);
      if (tr) {
        const a = 0.65;
        const nvx = dt ? (r.cx - tr.cx) / dt : 0, nvy = dt ? (r.cy - tr.cy) / dt : 0;
        tr.vx = tr.vx * 0.5 + nvx * 0.5; tr.vy = tr.vy * 0.5 + nvy * 0.5;
        tr.cx = tr.cx * (1 - a) + r.cx * a; tr.cy = tr.cy * (1 - a) + r.cy * a;
        const areaChange = Math.abs(Math.log(r.area / tr.descArea));
        // Descriptor cache: refresh only every few frames or on big change.
        if (frameNo - tr.descFrame >= 6 || areaChange > 0.3) {
          tr.desc = describeRegion(r, planes, w, h, s.regionMap, refL);
          tr.descFrame = frameNo; tr.descArea = r.area; tr.dv++;
          described++;
        }
        tr.area = r.area; tr.outline = r.outline; tr.hits++; tr.missed = 0;
        tr.mL = r.mL; tr.mA = r.mA; tr.mB = r.mB;
        tr.box = [r.minX, r.minY, r.maxX, r.maxY];
        tr.edge = r.touchesEdge;
        tr.lastSeen = t;
      } else {
        tr = {
          id: nextId++, cx: r.cx, cy: r.cy, vx: gvx, vy: gvy, area: r.area,
          outline: r.outline, desc: describeRegion(r, planes, w, h, s.regionMap, refL),
          descFrame: frameNo, descArea: r.area, dv: 1, hits: 1, missed: 0, lastSeen: t,
          mL: r.mL, mA: r.mA, mB: r.mB, box: [r.minX, r.minY, r.maxX, r.maxY], edge: r.touchesEdge,
        };
        described++;
        tracks.set(tr.id, tr);
      }
      tr.region = r;
      seen.add(tr.id);
    }
    for (const tr of tracks.values()) {
      if (seen.has(tr.id)) continue;
      tr.region = null;
      tr.missed++;
      // Coast with last velocity, easing toward camera motion — don't snap to zero.
      tr.vx = (tr.vx || 0) * 0.85 + gvx * 0.15;
      tr.vy = (tr.vy || 0) * 0.85 + gvy * 0.15;
      const dx = tr.vx * dt, dy = tr.vy * dt;
      tr.cx += dx; tr.cy += dy;
      if (tr.box) tr.box = [tr.box[0] + dx, tr.box[1] + dy, tr.box[2] + dx, tr.box[3] + dy];
      const farOff = tr.cx < -40 || tr.cy < -40 || tr.cx > w + 40 || tr.cy > h + 40;
      const age = t - (tr.lastSeen || 0);
      if (farOff && tr.missed > 6) tracks.delete(tr.id);
      else if (tr.missed > TRACK_HOLD_MISSES || age > TRACK_HOLD_MS) tracks.delete(tr.id);
    }
    const t2 = now();

    // Taps / scans / an ML mask set the reference sock.
    let scan = null;
    if (o.refMask) scan = refFromMask(o.refMask, o.excludeId);
    if (o.tap) scan = refFromTap(o.tap);
    if (o.scanBox) scan = refFromBox(o.scanBox, refL);
    if (scan && scan.ok && ref) scan.appearance = appearanceOf(ref.desc);

    const thr = thresholdFor(o.strictness ?? 0.55);
    const minHits = o.still ? 1 : 3;
    const scanMode = (o.mode || 'scan') === 'scan';
    // Auto Pairs still ignores tiny crumbs and huge edge blobs. Scan mode
    // scores any current region by appearance — including smaller, partial,
    // or bunched socks — and does not gate on area relative to the scan.
    const areas = [...tracks.values()].map((x) => x.area).sort((a, b) => a - b);
    const medA = areas.length ? areas[areas.length >> 1] : 0;
    const minA = Math.max(n * 0.003, medA * 0.3);
    const y0 = (o.roi?.y0 ?? 0), y1 = (o.roi?.y1 ?? h);
    const live = [...tracks.values()].filter((x) => {
      const sticky = isSticky(x.id);
      // Only currently-detected socks are scored; coasting tracks keep a
      // shown lock via match/pair hold, without refreshing lastHit.
      if (x.missed > 0) return false;
      if (x.hits < minHits && !sticky) return false;
      if (!scanMode && x.area < minA && !sticky) return false;
      if (!sticky && (x.cy < y0 || x.cy > y1 || x.edge)) return false;
      return true;
    });
    let pairs = [], matches = [];
    if ((o.mode || 'scan') === 'pairs') pairs = updatePairs(live, thr, !!o.still);
    else if (ref) matches = updateMatches(live, thr, !!o.still);
    const t3 = now();

    const show = new Set();
    for (const p of pairs) { show.add(p.a); show.add(p.b); }
    for (const m of matches) show.add(m.id);
    if (ref && ref.excludeId) show.add(ref.excludeId);
    const outTracks = [];
    for (const tr of tracks.values()) {
      const o2 = { id: tr.id, cx: tr.cx, cy: tr.cy, vx: tr.vx, vy: tr.vy, area: tr.area, missed: tr.missed, hits: tr.hits, box: tr.box, dv: tr.dv };
      if ((show.has(tr.id) || tr.missed === 0) && tr.outline) o2.outline = Array.from(tr.outline);
      if (ref && tr.id !== ref.excludeId) o2.colorScore = refSim(tr);
      outTracks.push(o2);
    }
    return {
      w, h, frame: frameNo,
      tracks: outTracks, pairs, matches, scan,
      ref: ref ? { excludeId: ref.excludeId || 0 } : null,
      regions: s.regions.length,
      timing: { seg: t1 - t0, track: t2 - t1, match: t3 - t2, total: t3 - t0, described },
    };
  }

  function setRef(desc, excludeId) {
    ref = { desc: appearanceOf(desc), excludeId };
    refVersion++;
    matchState = new Map();
  }

  function setAppearance(desc, excludeId) {
    if (!desc) return { ok: false };
    setRef(desc, excludeId || 0);
    return { ok: true };
  }

  function getAppearance() {
    return ref ? { desc: appearanceOf(ref.desc), excludeId: ref.excludeId || 0 } : null;
  }

  /** Build a reference descriptor from a binary mask (row-major, 1 = sock). */
  function refFromMask(mask, excludeId) {
    if (!mask || mask.length !== W * H) return { ok: false };
    const px = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) px.push(i);
    if (px.length < Math.max(24, (W * H) * 0.001)) return { ok: false };
    const r = regionLike(px);
    const map = new Int32Array(W * H).fill(-1);
    for (const i of px) map[i] = 0;
    const refL = lastSeg?.surface?.L ?? 70;
    const desc = describeRegion(r, planes, W, H, map, refL);
    let ex = excludeId || 0;
    if (!ex) {
      for (const tr of tracks.values()) {
        if (tr.cx >= r.minX && tr.cx <= r.maxX && tr.cy >= r.minY && tr.cy <= r.maxY && tr.area > px.length * 0.3) ex = tr.id;
      }
    }
    setRef(desc, ex);
    return { ok: true, method: 'mask', box: [r.minX, r.minY, r.maxX, r.maxY], id: ex };
  }

  function refFromTap(p) {
    const s = lastSeg;
    const x = Math.round(p.x), y = Math.round(p.y);
    let rid = -1;
    if (x >= 0 && y >= 0 && x < W && y < H) rid = s.regionMap[y * W + x];
    if (rid < 0) {
      // Nearest region centre within a finger's reach.
      let bd = Math.hypot(W, H) * 0.06;
      for (const r of s.regions) {
        const d = Math.hypot(r.cx - p.x, r.cy - p.y);
        if (d < bd) { bd = d; rid = r.id; }
      }
    }
    if (rid < 0) return { ok: false };
    const tr = [...tracks.values()].find((t) => t.region && t.region.id === rid);
    if (!tr) return { ok: false };
    setRef(tr.desc, tr.id);
    return { ok: true, method: 'tap', box: tr.box, id: tr.id };
  }

  function refFromBox(b, refL) {
    const s = lastSeg;
    const x0 = Math.max(0, Math.floor(b.x0)), y0 = Math.max(0, Math.floor(b.y0));
    const x1 = Math.min(W - 1, Math.ceil(b.x1)), y1 = Math.min(H - 1, Math.ceil(b.y1));
    const boxArea = (x1 - x0 + 1) * (y1 - y0 + 1);
    // 1) A detected sock mostly inside the target.
    let best = null, bestScore = 0;
    for (const r of s.regions) {
      if (r.cx < x0 || r.cx > x1 || r.cy < y0 || r.cy > y1) continue;
      const score = r.area * (1 - 0.5 * Math.hypot(r.cx - (x0 + x1) / 2, r.cy - (y0 + y1) / 2) / Math.hypot(x1 - x0, y1 - y0));
      if (score > bestScore) { bestScore = score; best = r; }
    }
    if (best && best.area > boxArea * 0.08) {
      const tr = [...tracks.values()].find((t) => t.region === best);
      if (tr) {
        setRef(tr.desc, tr.id);
        return { ok: true, method: 'region', box: tr.box, id: tr.id };
      }
    }
    // 2) Held-up sock: pixels that differ from the target's border ring.
    const { Ls, As, Bs } = planes;
    const ring = [];
    for (let x = x0; x <= x1; x += 2) { ring.push(y0 * W + x, y1 * W + x); }
    for (let y = y0; y <= y1; y += 2) { ring.push(y * W + x0, y * W + x1); }
    const med = (arr) => arr.slice().sort((p, q) => p - q)[arr.length >> 1];
    const rl = med(ring.map((i) => Ls[i])), ra = med(ring.map((i) => As[i])), rb = med(ring.map((i) => Bs[i]));
    const map = new Int32Array(W * H).fill(-1);
    const inBox = new Uint8Array(W * H);
    for (let y = y0 + 1; y < y1; y++) {
      for (let x = x0 + 1; x < x1; x++) {
        const i = y * W + x;
        const d = Math.hypot((Ls[i] - rl) * 0.5, As[i] - ra, Bs[i] - rb);
        if (d > 11) inBox[i] = 1;
      }
    }
    let bestPx = null;
    const seenPx = new Uint8Array(W * H);
    for (let y = y0 + 1; y < y1; y++) {
      for (let x = x0 + 1; x < x1; x++) {
        const s0 = y * W + x;
        if (!inBox[s0] || seenPx[s0]) continue;
        const q = [s0]; seenPx[s0] = 1;
        for (let k = 0; k < q.length; k++) {
          const i = q[k];
          for (const j of [i - 1, i + 1, i - W, i + W]) {
            if (j < 0 || j >= W * H || !inBox[j] || seenPx[j]) continue;
            seenPx[j] = 1; q.push(j);
          }
        }
        if (!bestPx || q.length > bestPx.length) bestPx = q;
      }
    }
    let method = 'held';
    if (!bestPx || bestPx.length < boxArea * 0.06) {
      // 3) Fall back to the inner part of the target.
      method = 'center';
      bestPx = [];
      const mx = (x1 - x0) * 0.2, my = (y1 - y0) * 0.2;
      for (let y = Math.round(y0 + my); y <= y1 - my; y++) for (let x = Math.round(x0 + mx); x <= x1 - mx; x++) bestPx.push(y * W + x);
    }
    const r = regionLike(bestPx);
    for (const i of bestPx) map[i] = 0;
    const desc = describeRegion(r, planes, W, H, map, refL);
    desc.held = true;
    // Exclude any tracked sock sitting where the scanned one is.
    let excludeId = 0;
    for (const tr of tracks.values()) {
      if (tr.cx >= x0 && tr.cx <= x1 && tr.cy >= y0 && tr.cy <= y1 && tr.area > bestPx.length * 0.4) excludeId = tr.id;
    }
    setRef(desc, excludeId);
    return { ok: true, method, box: [r.minX, r.minY, r.maxX, r.maxY], id: excludeId };
  }

  function regionLike(px) {
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, minX = W, minY = H, maxX = 0, maxY = 0;
    for (const i of px) {
      const x = i % W, y = (i / W) | 0;
      sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    const a = px.length, cx = sx / a, cy = sy / a;
    const cxx = sxx / a - cx * cx, cyy = syy / a - cy * cy, cxy = sxy / a - cx * cy;
    const tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const ev1 = tr / 2 + disc, ev2 = Math.max(1e-3, tr / 2 - disc);
    return {
      id: 0, pixels: px, area: a, cx, cy, minX, minY, maxX, maxY,
      angle: 0.5 * Math.atan2(2 * cxy, cxx - cyy), major: Math.sqrt(ev1), minor: Math.sqrt(ev2),
      elongation: Math.sqrt(ev1 / ev2),
    };
  }

  function updatePairs(live, thr, still) {
    const n = live.length;
    const m = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const s = sim(live[i], live[j]);
        m[i * n + j] = s; m[j * n + i] = s;
      }
    }
    const ids = live.map((t) => t.id);
    const found = confidentPairs(ids, m, thr * 0.92);
    const alive = new Set(ids);
    const foundKeys = new Set();
    for (const p of found) {
      const key = `${p.a}:${p.b}`;
      foundKeys.add(key);
      let st = pairState.get(key);
      if (!st) {
        st = { a: p.a, b: p.b, ema: still ? p.score : p.score * 0.7, shown: false, slot: 0, count: still ? 9 : 0, missed: 0, lastHit: lastT };
        pairState.set(key, st);
      } else st.ema = st.ema * 0.6 + p.score * 0.4;
      st.count++;
      st.score = p.score;
      const seenNow = still || ((tracks.get(p.a)?.missed || 0) === 0 && (tracks.get(p.b)?.missed || 0) === 0);
      if (seenNow && p.score >= thr - DROP_SLACK) { st.missed = 0; st.lastHit = lastT; }
      else st.missed = (st.missed || 0) + 1;
    }
    for (const [key, st] of pairState) {
      if (!foundKeys.has(key)) {
        st.ema *= still ? 0.82 : MISS_EMA_DECAY;
        st.missed = (st.missed || 0) + 1;
      }
      const gone = !tracks.has(st.a) || !tracks.has(st.b);
      if (still && (gone || st.ema < thr * 0.6)) pairState.delete(key);
      else if (holdExpired(st) && (gone || st.ema < thr * 0.45)) pairState.delete(key);
    }
    return selectShown([...pairState.values()], thr, (st) => [st.a, st.b], alive, 0, 3);
  }

  function updateMatches(live, thr, still) {
    const found = [];
    for (const t of live) {
      if (ref.excludeId && t.id === ref.excludeId) continue;
      found.push({ id: t.id, s: refSim(t) });
    }
    found.sort((a, b) => b.s - a.s);
    const top = found.slice(0, 6);
    const topIds = new Set(top.map((f) => f.id));
    for (const f of top) {
      let st = matchState.get(f.id);
      if (!st) {
        st = { id: f.id, ema: still ? f.s : f.s * 0.75, shown: false, slot: 0, count: still ? 9 : 0, missed: 0, lastHit: lastT };
        matchState.set(f.id, st);
      } else st.ema = st.ema * 0.6 + f.s * 0.4;
      st.count++;
      st.score = f.s;
      const seenNow = still || (tracks.get(f.id)?.missed || 0) === 0;
      if (seenNow && f.s >= thr - DROP_SLACK) { st.missed = 0; st.lastHit = lastT; }
      else st.missed = (st.missed || 0) + 1;
    }
    for (const [id, st] of matchState) {
      if (!topIds.has(id)) {
        st.ema *= still ? 0.85 : MISS_EMA_DECAY;
        st.missed = (st.missed || 0) + 1;
      }
      const gone = !tracks.has(id);
      if (still && (gone || st.ema < thr * 0.6)) matchState.delete(id);
      else if (holdExpired(st) && (gone || st.ema < thr * 0.45)) matchState.delete(id);
    }
    const alive = new Set(live.map((t) => t.id));
    const states = [...matchState.values()];
    const bestEma = states.reduce((a, s) => Math.max(a, s.ema), 0);
    return selectShown(states, thr, (st) => [st.id], alive, bestEma - 0.1, 2)
      .map((st) => ({ id: st.id, score: st.ema, slot: st.slot }));
  }

  /** Hysteresis + max-3 selection with stable slots. Drop bar is lower than acquire, and a shown lock is held ~2s through misses. */
  function selectShown(states, thr, members, alive, floor = 0, minCount = 1) {
    const on = Math.max(thr, floor), off = thr - DROP_SLACK;
    for (const st of states) {
      const ok = members(st).every((id) => tracks.has(id));
      if (!st.shown) continue;
      // Still/photo frames are a fresh snapshot; live camera holds through brief misses.
      if (st.ema < off || !ok) {
        if (!holdExpired(st) && lastT) continue;
        st.shown = false; st.slot = 0;
      }
    }
    const shown = states.filter((s) => s.shown).sort((a, b) => b.ema - a.ema);
    const used = new Set();
    for (const s of shown) for (const id of members(s)) used.add(id);
    const cands = states.filter((s) => !s.shown && s.ema >= on && (s.count || 0) >= minCount && members(s).every((id) => alive.has(id)))
      .sort((a, b) => b.ema - a.ema);
    for (const c of cands) {
      if (members(c).some((id) => used.has(id))) continue;
      if (shown.length >= MAX_SHOWN) {
        const weakest = shown[shown.length - 1];
        if (c.ema < weakest.ema + 0.05) continue;
        weakest.shown = false; weakest.slot = 0;
        for (const id of members(weakest)) used.delete(id);
        shown.pop();
      }
      c.shown = true;
      shown.push(c);
      shown.sort((a, b) => b.ema - a.ema);
      for (const id of members(c)) used.add(id);
    }
    const slots = new Set(shown.map((s) => s.slot).filter(Boolean));
    for (const s of shown) {
      if (s.slot) continue;
      let k = 1; while (slots.has(k)) k++;
      s.slot = k; slots.add(k);
    }
    return shown.map((s) => ({ ...s }));
  }

  /** Debug/test: every confident pair among current tracks, no display cap. */
  function allPairs(strictness = 0.55) {
    const live = [...tracks.values()];
    const n = live.length;
    const m = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const s = sim(live[i], live[j]); m[i * n + j] = s; m[j * n + i] = s; }
    return confidentPairs(live.map((t) => t.id), m, thresholdFor(strictness));
  }

  function simBetween(idA, idB) {
    const a = tracks.get(idA), b = tracks.get(idB);
    return a && b ? sim(a, b) : 0;
  }

  return { process, reset, clearRef, allPairs, simBetween, getTracks: () => tracks, hasRef: () => !!ref, setRefFromMask: (mask, excludeId) => refFromMask(mask, excludeId), setAppearance, getAppearance };
}
