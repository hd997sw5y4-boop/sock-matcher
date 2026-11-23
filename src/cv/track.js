/**
 * Temporal smoothing so pair numbers and boxes don't flicker in live video.
 */
export function createTracker() {
  let nextId = 1;
  let tracks = [];
  let pairLabels = new Map(); // key "idA-idB" (sorted) -> number
  let nextPair = 1;
  let stale = 0;

  function reset() {
    nextId = 1;
    tracks = [];
    pairLabels = new Map();
    nextPair = 1;
    stale = 0;
  }

  function update(blobs, descs, pairing, workW, workH) {
    const distGate = Math.hypot(workW, workH) * 0.12;
    const usedTracks = new Set();
    const usedBlobs = new Set();
    const assign = new Map(); // blobIndex -> track

    const cands = [];
    for (let i = 0; i < blobs.length; i++) {
      for (const t of tracks) {
        const d = Math.hypot(blobs[i].cx - t.cx, blobs[i].cy - t.cy);
        if (d > distGate * (t.missed ? 1.4 : 1)) continue;
        cands.push({ i, t, d });
      }
    }
    cands.sort((a, b) => a.d - b.d);
    for (const c of cands) {
      if (usedBlobs.has(c.i) || usedTracks.has(c.t.id)) continue;
      usedBlobs.add(c.i);
      usedTracks.add(c.t.id);
      assign.set(c.i, c.t);
    }

    const alpha = 0.45;
    const live = [];
    for (let i = 0; i < blobs.length; i++) {
      const b = blobs[i];
      let t = assign.get(i);
      if (!t) {
        t = {
          id: nextId++,
          cx: b.cx, cy: b.cy,
          minX: b.minX, minY: b.minY, maxX: b.maxX, maxY: b.maxY,
          contour: b.contour,
          desc: descs[i],
          age: 0, missed: 0, confirmed: 1,
        };
      } else {
        t.cx = t.cx * (1 - alpha) + b.cx * alpha;
        t.cy = t.cy * (1 - alpha) + b.cy * alpha;
        t.minX = t.minX * (1 - alpha) + b.minX * alpha;
        t.minY = t.minY * (1 - alpha) + b.minY * alpha;
        t.maxX = t.maxX * (1 - alpha) + b.maxX * alpha;
        t.maxY = t.maxY * (1 - alpha) + b.maxY * alpha;
        t.contour = b.contour;
        t.desc = descs[i];
        t.missed = 0;
        t.confirmed++;
      }
      t.blobIndex = i;
      t.age++;
      live.push(t);
    }

    for (const t of tracks) {
      if (usedTracks.has(t.id)) continue;
      t.missed++;
      if (t.missed <= 4) {
        t.blobIndex = -1;
        live.push(t);
      }
    }
    tracks = live;

    // Stable pair numbers from track ids.
    const idOfBlob = (idx) => {
      const t = live.find((x) => x.blobIndex === idx);
      return t ? t.id : -1;
    };
    const current = new Map();
    const usedNums = new Set();
    for (const p of pairing.pairs) {
      const idA = idOfBlob(p.a);
      const idB = idOfBlob(p.b);
      if (idA < 0 || idB < 0) continue;
      const key = idA < idB ? `${idA}-${idB}` : `${idB}-${idA}`;
      let num = pairLabels.get(key);
      if (!num) {
        // Reuse a number if one of the tracks kept its old partner loosely? no — new number.
        num = nextPair++;
        pairLabels.set(key, num);
      }
      current.set(key, { num, a: p.a, b: p.b, score: p.score, idA, idB });
      usedNums.add(num);
    }

    // Drop labels that have been gone for a few frames so numbers don't explode.
    stale++;
    if (stale % 3 === 0) {
      for (const key of [...pairLabels.keys()]) {
        if (!current.has(key)) pairLabels.delete(key);
      }
    }

    const socks = blobs.map((b, i) => {
      const t = live.find((x) => x.blobIndex === i);
      return {
        index: i,
        trackId: t ? t.id : -1,
        cx: t ? t.cx : b.cx,
        cy: t ? t.cy : b.cy,
        minX: t ? t.minX : b.minX,
        minY: t ? t.minY : b.minY,
        maxX: t ? t.maxX : b.maxX,
        maxY: t ? t.maxY : b.maxY,
        contour: b.contour,
        desc: descs[i],
        area: b.area,
      };
    });

    const pairs = [...current.values()].sort((a, b) => a.num - b.num);
    const pairedIdx = new Set();
    for (const p of pairs) { pairedIdx.add(p.a); pairedIdx.add(p.b); }
    const unmatched = socks.filter((s) => !pairedIdx.has(s.index));

    return { socks, pairs, unmatched };
  }

  return { update, reset };
}
