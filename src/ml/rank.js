/**
 * Combine a visual-pattern embedding with the classical color/stripe score,
 * and pick a small number of confident matches. Pure JS, no model needed.
 */

export const MAX_SHOWN = 3;
export const ACQUIRE = 0.62;
export const DROP = 0.54;
export const MARGIN = 0.04;

/** Cosine of two L2-normalized vectors. */
export function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function l2norm(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s) || 1;
  const o = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) o[i] = v[i] / n;
  return o;
}

/**
 * Map a DINOv2 cosine (~0.45–0.95) and a classical 0–1 color score
 * onto a single 0–1 confidence. Pattern evidence leads; color/stripe
 * bands still matter for heel, toe, cuff and stripe count.
 */
export function combinedScore(cosineSim, colorSim = 0) {
  // Observed mates on real piles often land around 0.55–0.92 cosine.
  const c = Math.max(0, Math.min(1, (cosineSim - 0.48) / 0.40));
  const col = Math.max(0, Math.min(1, colorSim));
  // Prints and logos lean on the embedding; solid colors still lean on the
  // classical bands. A strong color mismatch vetoes a pattern-only glow.
  let s = Math.max(0.7 * c + 0.3 * col, 0.3 * c + 0.7 * col);
  if (col < 0.38) s *= 0.72;
  return s;
}

/**
 * Keep only the top 1–3 candidates that clear the acquire bar and are
 * not a near-tie with a crowd of weaker ones. Empty when unsure.
 */
export function pickMatches(cands, {
  acquire = ACQUIRE,
  maxShown = MAX_SHOWN,
  margin = MARGIN,
} = {}) {
  const sorted = cands
    .filter((x) => x && Number.isFinite(x.score))
    .slice()
    .sort((a, b) => b.score - a.score);
  if (!sorted.length || sorted[0].score < acquire) return [];
  const out = [];
  for (const c of sorted) {
    if (out.length >= maxShown) break;
    if (c.score < acquire) break;
    if (out.length && out[0].score - c.score > 0.14) break;
    // Fourth-place (or the next leftover) must lose by a margin, else we're unsure.
    const rival = sorted[out.length + 1];
    if (rival && c.score - rival.score < margin && c.score < acquire + 0.06) continue;
    out.push(c);
  }
  return out;
}

/**
 * Light temporal hysteresis for live highlights. A match lights after a
 * couple of good hits and drops after a few misses (~0.4 s), so brief
 * flicker is smoothed without a wrong glow sticking around.
 */
export function createScoreTracker({
  acquire = ACQUIRE,
  drop = DROP,
  holdMs = 400,
  minHits = 2,
  maxShown = MAX_SHOWN,
  margin = MARGIN,
} = {}) {
  const state = new Map();

  function update(cands, now) {
    const seen = new Set();
    for (const c of cands) {
      const id = c.id;
      seen.add(id);
      let st = state.get(id);
      if (!st) {
        st = { id, ema: c.score, hits: 1, missed: 0, lastHit: now, shown: false, slot: 0, score: c.score, cosine: c.cosine, color: c.color };
        state.set(id, st);
      } else {
        st.ema = st.ema * 0.55 + c.score * 0.45;
        st.hits++;
        st.missed = 0;
        st.lastHit = now;
        st.score = c.score;
        st.cosine = c.cosine;
        st.color = c.color;
        if (c.a != null) { st.a = c.a; st.b = c.b; }
      }
      if (c.a != null && st.a == null) { st.a = c.a; st.b = c.b; }
    }
    for (const [id, st] of state) {
      if (!seen.has(id)) {
        st.ema *= 0.9;
        st.missed++;
      }
      const expired = st.missed >= 5 && now - (st.lastHit || 0) >= holdMs;
      if (st.shown && (st.ema < drop || expired)) { st.shown = false; st.slot = 0; }
      if (!st.shown && expired && st.ema < drop) state.delete(id);
    }

    const live = [...state.values()];
    for (const st of live) {
      if (st.shown && st.ema < drop) { st.shown = false; st.slot = 0; }
    }
    const shown = live.filter((s) => s.shown).sort((a, b) => b.ema - a.ema);
    const used = new Set();
    for (const s of shown) {
      used.add(s.id);
      if (s.a != null) { used.add(s.a); used.add(s.b); }
    }
    const picked = pickMatches(
      live.filter((s) => !s.shown && s.hits >= minHits).map((s) => ({ ...s, score: s.ema })),
      { acquire, maxShown: maxShown, margin },
    );
    for (const c of picked) {
      const st = state.get(c.id);
      if (!st || used.has(st.id) || (st.a != null && (used.has(st.a) || used.has(st.b)))) continue;
      if (shown.length >= maxShown) {
        const weakest = shown[shown.length - 1];
        if (st.ema < weakest.ema + 0.05) continue;
        weakest.shown = false; weakest.slot = 0;
        used.delete(weakest.id);
        if (weakest.a != null) { used.delete(weakest.a); used.delete(weakest.b); }
        shown.pop();
      }
      st.shown = true;
      shown.push(st);
      shown.sort((a, b) => b.ema - a.ema);
      used.add(st.id);
      if (st.a != null) { used.add(st.a); used.add(st.b); }
    }
    const slots = new Set(shown.map((s) => s.slot).filter(Boolean));
    for (const s of shown) {
      if (s.slot) continue;
      let k = 1; while (slots.has(k)) k++;
      s.slot = k; slots.add(k);
    }
    return shown.slice(0, maxShown).map((s) => ({
      id: s.id, a: s.a, b: s.b, score: s.ema, cosine: s.cosine, color: s.color, slot: s.slot,
    }));
  }

  function reset() { state.clear(); }

  return { update, reset };
}
