/**
 * Sock segmentation on a small downscaled frame.
 *
 * 1. Surface model: the dominant a*b* chroma of the frame is the surface
 *    (bed sheet, floor). Its lightness and chroma are estimated per coarse
 *    grid cell, so wrinkles, vignetting and soft shadows don't become socks.
 * 2. Foreground = pixels whose chroma or lightness leaves the local surface.
 * 3. Morphological open/close, connected components.
 * 4. Big merged components (touching socks) are split by color-aware region
 *    growing, then tiny fragments are merged back into neighbours.
 * All buffers are reused between frames.
 */

const AB_RANGE = 64; // a,b in [-64, 64)
const AB_BIN = 2;
const AB_N = (AB_RANGE * 2) / AB_BIN; // 64

export function createSegmenter() {
  let w = 0, h = 0, n = 0;
  let fg, tmp8, label, sub, queue, order, bgL, bgA, bgB;
  const abHist = new Float32Array(AB_N * AB_N);
  const abSmooth = new Float32Array(AB_N * AB_N);

  function ensure(W, H) {
    if (W === w && H === h) return;
    w = W; h = H; n = w * h;
    fg = new Uint8Array(n);
    tmp8 = new Uint8Array(n);
    label = new Int32Array(n);
    sub = new Int32Array(n);
    queue = new Int32Array(n);
    order = new Int32Array(n);
    bgL = new Float32Array(n);
    bgA = new Float32Array(n);
    bgB = new Float32Array(n);
  }

  /** Fill the smoothed a*b* histogram. Returns nothing; reads Ls/As/Bs-style arrays. */
  function fillAbHist(A, B, L) {
    abHist.fill(0);
    for (let i = 0; i < n; i += 2) {
      if (L[i] < 18) continue; // ignore very dark gaps
      const ai = ((A[i] + AB_RANGE) / AB_BIN) | 0;
      const bi = ((B[i] + AB_RANGE) / AB_BIN) | 0;
      if (ai < 0 || bi < 0 || ai >= AB_N || bi >= AB_N) continue;
      abHist[ai * AB_N + bi] += 1;
    }
    for (let i = 1; i < AB_N - 1; i++) {
      for (let j = 1; j < AB_N - 1; j++) {
        let s = 0;
        for (let di = -1; di <= 1; di++) {
          const row = (i + di) * AB_N;
          s += abHist[row + j - 1] + abHist[row + j] * 2 + abHist[row + j + 1];
        }
        s += abHist[i * AB_N + j] * 2;
        abSmooth[i * AB_N + j] = s;
      }
    }
  }

  function binToChroma(i, j) {
    return [i * AB_BIN - AB_RANGE + AB_BIN / 2, j * AB_BIN - AB_RANGE + AB_BIN / 2];
  }

  /** Dominant surface chroma (peak of a smoothed a*b* histogram). */
  function surfaceChroma(A, B, L) {
    fillAbHist(A, B, L);
    let best = -1, bi0 = AB_N / 2, bj0 = AB_N / 2;
    for (let i = 1; i < AB_N - 1; i++) {
      for (let j = 1; j < AB_N - 1; j++) {
        const s = abSmooth[i * AB_N + j];
        if (s > best) { best = s; bi0 = i; bj0 = j; }
      }
    }
    return binToChroma(bi0, bj0);
  }

  /** Top local-maxima chroma modes, well separated in a*b*. First entry is the global peak. */
  function chromaPeaks(A, B, L, k = 3) {
    fillAbHist(A, B, L);
    const peaks = [];
    for (let i = 1; i < AB_N - 1; i++) {
      for (let j = 1; j < AB_N - 1; j++) {
        const s = abSmooth[i * AB_N + j];
        if (s <= 0) continue;
        let isMax = true;
        for (let di = -1; di <= 1 && isMax; di++) {
          for (let dj = -1; dj <= 1; dj++) {
            if (!di && !dj) continue;
            if (abSmooth[(i + di) * AB_N + (j + dj)] > s) { isMax = false; break; }
          }
        }
        if (isMax) peaks.push({ i, j, s });
      }
    }
    peaks.sort((a, b) => b.s - a.s);
    const picked = [];
    for (const p of peaks) {
      if (picked.some((q) => Math.abs(q.i - p.i) + Math.abs(q.j - p.j) < 4)) continue;
      picked.push(p);
      if (picked.length >= k) break;
    }
    if (!picked.length) return [surfaceChroma(A, B, L)];
    return picked.map((p) => binToChroma(p.i, p.j));
  }

  /** Cheap quality of a surface guess: more mid-sized components, no giant blob. */
  function surfaceScore(Ls, As, Bs, sa, sb, override, sens) {
    const surface = surfaceModel(Ls, As, Bs, sa, sb, override);
    foreground(Ls, As, Bs, sens);
    cleanup();
    const comps = components();
    const minArea = Math.max(30, n * 0.0016);
    const midMax = n * 0.15;
    let mid = 0, largest = 0, fgN = 0;
    for (let i = 0; i < n; i++) if (fg[i]) fgN++;
    for (const c of comps) {
      if (c.area > largest) largest = c.area;
      if (c.area >= minArea && c.area <= midMax) mid++;
    }
    let s = mid * 2;
    if (largest > n * 0.40) s -= 24;
    else if (largest > n * 0.25) s -= 8;
    const frac = fgN / n;
    if (frac > 0.88) s -= 6;
    if (frac < 0.12) s -= 6;
    return { score: s, mid, largest, fgFrac: frac, comps, surface };
  }

  /**
   * Local surface model on a coarse grid: median L and mean a,b of pixels
   * near the surface chroma; empty cells borrow from neighbours.
   */
  function surfaceModel(L, A, B, sa, sb, override) {
    const gx = 6;
    const gy = Math.max(4, Math.round(gx * h / w));
    const cells = gx * gy;
    const lh = new Uint16Array(cells * 26); // L histogram, 4-unit bins
    const sumA = new Float32Array(cells), sumB = new Float32Array(cells), cnt = new Float32Array(cells);
    const cw = w / gx, ch = h / gy;
    const tc2 = 9 * 9;
    for (let y = 0; y < h; y += 2) {
      const cy = Math.min(gy - 1, (y / ch) | 0);
      for (let x = 0; x < w; x += 2) {
        const i = y * w + x;
        const l = L[i];
        if (l < 15) continue;
        // Shadows of the surface keep hue but lose chroma in proportion.
        const da = A[i] - sa, db = B[i] - sb;
        if (da * da + db * db > tc2) continue;
        const c = cy * gx + Math.min(gx - 1, (x / cw) | 0);
        lh[c * 26 + Math.min(25, (l / 4) | 0)]++;
        sumA[c] += A[i]; sumB[c] += B[i]; cnt[c]++;
      }
    }
    const cL = new Float32Array(cells), cA = new Float32Array(cells), cB = new Float32Array(cells);
    const ok = new Uint8Array(cells);
    const perCell = (cw * ch) / 4;
    let gL = 0, gN = 0;
    for (let c = 0; c < cells; c++) {
      if (cnt[c] < perCell * 0.12) continue;
      // Mode of the L histogram: the surface is the most common lightness in
      // the cell, even when white socks share its chroma.
      let k = 0, bestV = -1;
      for (let q = 0; q < 26; q++) {
        const v = lh[c * 26 + q] * 2 + (q > 0 ? lh[c * 26 + q - 1] : 0) + (q < 25 ? lh[c * 26 + q + 1] : 0);
        if (v > bestV) { bestV = v; k = q; }
      }
      cL[c] = k * 4 + 2; cA[c] = sumA[c] / cnt[c]; cB[c] = sumB[c] / cnt[c];
      ok[c] = 1; gL += cL[c]; gN++;
    }
    const globalL = gN ? gL / gN : 60;
    if (override) {
      for (let c = 0; c < cells; c++) {
        if (!ok[c]) { cL[c] = override.L; cA[c] = override.a; cB[c] = override.b; ok[c] = 1; }
      }
    }
    // Fill holes from neighbours.
    for (let pass = 0; pass < gx + gy; pass++) {
      let missing = 0;
      for (let y = 0; y < gy; y++) {
        for (let x = 0; x < gx; x++) {
          const c = y * gx + x;
          if (ok[c]) continue;
          let s = 0, sl = 0, sa2 = 0, sb2 = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx, yy = y + dy;
              if (xx < 0 || yy < 0 || xx >= gx || yy >= gy) continue;
              const cc = yy * gx + xx;
              if (ok[cc] !== 1) continue;
              s++; sl += cL[cc]; sa2 += cA[cc]; sb2 += cB[cc];
            }
          }
          if (s) { cL[c] = sl / s; cA[c] = sa2 / s; cB[c] = sb2 / s; ok[c] = 2; } else missing++;
        }
      }
      for (let c = 0; c < cells; c++) if (ok[c] === 2) ok[c] = 1;
      if (!missing) break;
    }
    for (let c = 0; c < cells; c++) {
      if (!ok[c]) { cL[c] = globalL; cA[c] = sa; cB[c] = sb; }
    }
    // Bilinear upsample to per-pixel surface planes.
    for (let y = 0; y < h; y++) {
      let fy = (y + 0.5) / ch - 0.5;
      fy = Math.max(0, Math.min(gy - 1, fy));
      const y0 = Math.floor(fy), y1 = Math.min(gy - 1, y0 + 1), ty = fy - y0;
      for (let x = 0; x < w; x++) {
        let fx = (x + 0.5) / cw - 0.5;
        fx = Math.max(0, Math.min(gx - 1, fx));
        const x0 = Math.floor(fx), x1 = Math.min(gx - 1, x0 + 1), tx = fx - x0;
        const c00 = y0 * gx + x0, c01 = y0 * gx + x1, c10 = y1 * gx + x0, c11 = y1 * gx + x1;
        const w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;
        const i = y * w + x;
        bgL[i] = cL[c00] * w00 + cL[c01] * w01 + cL[c10] * w10 + cL[c11] * w11;
        bgA[i] = cA[c00] * w00 + cA[c01] * w01 + cA[c10] * w10 + cA[c11] * w11;
        bgB[i] = cB[c00] * w00 + cB[c01] * w01 + cB[c10] * w10 + cB[c11] * w11;
      }
    }
    return { L: globalL, a: sa, b: sb };
  }

  function foreground(L, A, B, sens) {
    // sens 0..1: higher = more pixels count as sock.
    const k = 1.35 - 0.7 * sens;
    const tc = 8.5 * k, tHi = 11 * k, tLo = 30 * k;
    for (let i = 0; i < n; i++) {
      const bl = bgL[i];
      const l = L[i];
      const ratio = Math.max(0.45, Math.min(1.15, l / Math.max(1, bl)));
      const da = A[i] - bgA[i] * ratio, db = B[i] - bgB[i] * ratio;
      const dl = l - bl;
      const sc = (da * da + db * db) / (tc * tc);
      const sl = dl > 0 ? (dl * dl) / (tHi * tHi) : (dl * dl) / (tLo * tLo);
      fg[i] = sc + sl > 1 ? 1 : 0;
    }
  }

  /** 3x3 erode/dilate using neighbour counts. */
  function morph3(src, dst, dilate) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let v;
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1) { dst[i] = src[i]; continue; }
        if (dilate) {
          v = src[i] | src[i - 1] | src[i + 1] | src[i - w] | src[i + w]
            | src[i - w - 1] | src[i - w + 1] | src[i + w - 1] | src[i + w + 1];
        } else {
          v = src[i] & src[i - 1] & src[i + 1] & src[i - w] & src[i + w];
        }
        dst[i] = v;
      }
    }
  }

  function cleanup() {
    morph3(fg, tmp8, false); // open
    morph3(tmp8, fg, true);
    morph3(fg, tmp8, true); // close
    morph3(tmp8, fg, false);
  }

  /** 4-connected components on fg. Pixels of comp k are order[start..start+area). */
  function components() {
    label.fill(-1);
    const comps = [];
    let write = 0;
    for (let s = 0; s < n; s++) {
      if (!fg[s] || label[s] >= 0) continue;
      const id = comps.length;
      const start = write;
      label[s] = id;
      order[write++] = s;
      let read = start;
      while (read < write) {
        const i = order[read++];
        const x = i % w;
        if (x > 0 && fg[i - 1] && label[i - 1] < 0) { label[i - 1] = id; order[write++] = i - 1; }
        if (x < w - 1 && fg[i + 1] && label[i + 1] < 0) { label[i + 1] = id; order[write++] = i + 1; }
        if (i >= w && fg[i - w] && label[i - w] < 0) { label[i - w] = id; order[write++] = i - w; }
        if (i < n - w && fg[i + w] && label[i + w] < 0) { label[i + w] = id; order[write++] = i + w; }
      }
      comps.push({ start, area: write - start });
    }
    return comps;
  }

  /**
   * Split one component by color region growing. Returns arrays of pixel
   * indices (as Int32Array views copied out).
   */
  function splitComponent(comp, Ls, As, Bs, minArea) {
    const { start, area } = comp;
    const compId = label[order[start]];
    for (let k = 0; k < area; k++) sub[order[start + k]] = -1;
    const regions = [];
    // Huge merged piles (socks glued through a mis-labelled floor) need a
    // tighter color gate so neighbouring prints come apart.
    const huge = area > n * 0.20;
    const te = huge ? 6.0 : 8, tr = huge ? 10 : 14;
    const te2 = te * te, tr2 = tr * tr;
    for (let k = 0; k < area; k++) {
      const s = order[start + k];
      if (sub[s] >= 0) continue;
      const rid = regions.length;
      let qh = 0, qt = 0;
      queue[qt++] = s; sub[s] = rid;
      let mL = Ls[s], mA = As[s], mB = Bs[s], cnt = 1;
      const px = [];
      while (qh < qt) {
        const i = queue[qh++];
        px.push(i);
        const x = i % w;
        const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, i - w, i + w];
        for (let t = 0; t < 4; t++) {
          const j = nb[t];
          if (j < 0 || j >= n || label[j] !== compId || sub[j] >= 0) continue;
          const dl = (Ls[j] - Ls[i]) * 0.6, da = As[j] - As[i], db = Bs[j] - Bs[i];
          if (dl * dl + da * da + db * db > te2) continue;
          const ml = (Ls[j] - mL / cnt) * 0.6, ma = As[j] - mA / cnt, mb = Bs[j] - mB / cnt;
          if (ml * ml + ma * ma + mb * mb > tr2) continue;
          sub[j] = rid;
          queue[qt++] = j;
          mL += Ls[j]; mA += As[j]; mB += Bs[j]; cnt++;
        }
      }
      regions.push({ px, L: mL / cnt, a: mA / cnt, b: mB / cnt, alive: true });
    }
    // Merge small fragments into the neighbour sharing the longest border.
    const small = Math.max(6, minArea * 0.6);
    for (let pass = 0; pass < 4; pass++) {
      let changed = false;
      const ordered = regions.map((r, i) => i).filter((i) => regions[i].alive && regions[i].px.length < small)
        .sort((p, q) => regions[p].px.length - regions[q].px.length);
      for (const rid of ordered) {
        const r = regions[rid];
        if (!r.alive || r.px.length >= small) continue;
        const border = new Map();
        for (const i of r.px) {
          const x = i % w;
          const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, i - w, i + w];
          for (const j of nb) {
            if (j < 0 || j >= n || label[j] !== compId) continue;
            const o = sub[j];
            if (o === rid || o < 0) continue;
            border.set(o, (border.get(o) || 0) + 1);
          }
        }
        let bestO = -1, bestC = 0;
        for (const [o, c] of border) {
          const q = regions[o];
          if (!q.alive || c <= bestC) continue;
          // Don't glue a differently colored object (e.g. a toy) onto a sock.
          const dl = (q.L - r.L) * 0.6, da = q.a - r.a, db = q.b - r.b;
          if (r.px.length > 4 && dl * dl + da * da + db * db > 26 * 26) continue;
          bestC = c; bestO = o;
        }
        if (bestO < 0) { r.alive = false; for (const i of r.px) sub[i] = -2; r.px = []; continue; }
        const t = regions[bestO];
        for (const i of r.px) { sub[i] = bestO; t.px.push(i); }
        r.alive = false; r.px = [];
        changed = true;
      }
      if (!changed) break;
    }
    // Merge neighbouring regions with very similar mean color (same sock, shading).
    const alive = regions.filter((r) => r.alive);
    for (const r of alive) {
      let sl = 0, sa = 0, sb = 0;
      for (const i of r.px) { sl += Ls[i]; sa += As[i]; sb += Bs[i]; }
      r.L = sl / r.px.length; r.a = sa / r.px.length; r.b = sb / r.px.length;
    }
    return regions.filter((r) => r.alive && r.px.length >= minArea).map((r) => r.px);
  }

  /** Region stats + shape from a pixel list. */
  function regionFromPixels(px, id) {
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    let minX = w, minY = h, maxX = 0, maxY = 0;
    for (let k = 0; k < px.length; k++) {
      const i = px[k];
      const x = i % w, y = (i / w) | 0;
      sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
    const a = px.length;
    const cx = sx / a, cy = sy / a;
    const cxx = sxx / a - cx * cx, cyy = syy / a - cy * cy, cxy = sxy / a - cx * cy;
    const tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const ev1 = tr / 2 + disc, ev2 = Math.max(1e-3, tr / 2 - disc);
    const angle = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
    const bw = maxX - minX + 1, bh = maxY - minY + 1;
    return {
      id, pixels: px, area: a, cx, cy, minX, minY, maxX, maxY,
      angle, major: Math.sqrt(ev1), minor: Math.sqrt(ev2),
      elongation: Math.sqrt(ev1 / ev2), solidity: a / (bw * bh),
      touchesEdge: minX <= 0 || minY <= 0 || maxX >= w - 1 || maxY >= h - 1,
    };
  }

  /** Star-shaped outline: farthest pixel per angle bin, smoothed. Relative to centroid. */
  function outline(r, bins = 40) {
    const rad = new Float32Array(bins);
    const { pixels: px, cx, cy } = r;
    for (let k = 0; k < px.length; k++) {
      const i = px[k];
      const dx = (i % w) - cx + 0.5 * Math.sign((i % w) - cx), dy = ((i / w) | 0) - cy + 0.5 * Math.sign(((i / w) | 0) - cy);
      const d2 = dx * dx + dy * dy;
      let ang = Math.atan2(dy, dx);
      const b = ((ang + Math.PI) / (2 * Math.PI) * bins | 0) % bins;
      if (d2 > rad[b]) rad[b] = d2;
    }
    const out = new Float32Array(bins * 2);
    const sm = new Float32Array(bins);
    for (let b = 0; b < bins; b++) sm[b] = Math.sqrt(rad[b]);
    // Fill empty bins and smooth.
    for (let b = 0; b < bins; b++) {
      if (sm[b] > 0) continue;
      let l = 1; while (l < bins && sm[(b - l + bins) % bins] === 0) l++;
      let rr = 1; while (rr < bins && sm[(b + rr) % bins] === 0) rr++;
      sm[b] = (sm[(b - l + bins) % bins] + sm[(b + rr) % bins]) / 2;
    }
    for (let b = 0; b < bins; b++) {
      const v = (sm[(b - 1 + bins) % bins] + 2 * sm[b] + sm[(b + 1) % bins]) / 4 + 1.2;
      const ang = (b + 0.5) / bins * 2 * Math.PI - Math.PI;
      out[b * 2] = Math.cos(ang) * v;
      out[b * 2 + 1] = Math.sin(ang) * v;
    }
    return out;
  }

  /**
   * @returns {{regions, surface}}
   */
  function segment(planes, W, H, opts = {}) {
    ensure(W, H);
    const { L, A, B, Ls, As, Bs } = planes;
    const sens = opts.sensitivity ?? 0.5;
    let sa, sb, surface, comps;
    if (opts.surface) {
      sa = opts.surface.a; sb = opts.surface.b;
      surface = surfaceModel(Ls, As, Bs, sa, sb, opts.surface);
      foreground(Ls, As, Bs, sens);
      cleanup();
      comps = components();
    } else {
      // If the pile dominates the frame, the global chroma peak is often the
      // laundry, not the floor. Socks then merge through the floor into one
      // giant blob that we have to throw away. When that happens, try the
      // next chroma modes and keep the one that yields mid-sized components.
      const peaks = chromaPeaks(As, Bs, Ls, 3);
      sa = peaks[0][0]; sb = peaks[0][1];
      let chosen = surfaceScore(Ls, As, Bs, sa, sb, null, sens);
      if (peaks.length > 1 && (chosen.mid < 8 || chosen.largest > n * 0.40)) {
        let bestAB = peaks[0];
        for (let p = 1; p < peaks.length; p++) {
          const alt = surfaceScore(Ls, As, Bs, peaks[p][0], peaks[p][1], null, sens);
          if (alt.score > chosen.score) { chosen = alt; bestAB = peaks[p]; }
        }
        sa = bestAB[0]; sb = bestAB[1];
        // Re-apply the winner so fg / surface buffers match chosen.comps.
        chosen = surfaceScore(Ls, As, Bs, sa, sb, null, sens);
      }
      surface = chosen.surface;
      comps = chosen.comps;
    }

    const minArea = Math.max(30, n * 0.0016);
    const maxArea = n * 0.3;
    const areas = comps.filter((c) => c.area >= minArea).map((c) => c.area).sort((a, b) => a - b);
    const med = areas.length ? areas[areas.length >> 1] : minArea * 4;
    // Cap the split trigger so a single dominant pile blob (when it's also
    // the median component) still gets color-split instead of being kept
    // whole and then discarded as > maxArea.
    const splitOver = Math.max(n * 0.02, Math.min(med * 2.2, n * 0.08));
    const regions = [];
    for (const c of comps) {
      if (c.area < minArea) continue;
      if (c.area > splitOver) {
        const parts = splitComponent(c, L, A, B, minArea);
        for (const px of parts) {
          if (px.length > maxArea) continue;
          regions.push(regionFromPixels(px, regions.length));
        }
      } else {
        const px = order.subarray(c.start, c.start + c.area).slice();
        regions.push(regionFromPixels(px, regions.length));
      }
    }
    const keep = regions.filter((r) => r.solidity > 0.18 && r.elongation < 7 && r.area <= maxArea);
    keep.forEach((r, i) => { r.id = i; r.outline = outline(r); });
    // Paint a region-id map (for the descriptor's interior test and hit tests).
    sub.fill(-1);
    for (const r of keep) for (let k = 0; k < r.pixels.length; k++) sub[r.pixels[k]] = r.id;
    return { regions: keep, surface, regionMap: sub, w, h, fg };
  }

  return { segment, surfaceAt: (i) => ({ L: bgL[i], a: bgA[i], b: bgB[i] }) };
}
