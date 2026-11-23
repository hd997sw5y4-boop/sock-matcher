/**
 * Overlay renderer. Runs every animation frame, independently of the vision
 * worker: tracked positions are extrapolated with their velocity and eased,
 * so outlines stay glued to socks between (slower) vision updates.
 */
export const SLOT_COLORS = ['#30D158', '#0A84FF', '#FF9F0A'];
const MATCH_COLOR = '#30D158';

export function createOverlay(canvas) {
  const ctx = canvas.getContext('2d');
  const items = new Map(); // key -> render item
  let drewLast = false;

  /**
   * targets: [{key, id, color, label, kind:'match'|'pair'|'ref', track}]
   * map: {toView(x,y) -> [vx,vy], scale}
   */
  function setTargets(targets, resultTime) {
    const keep = new Set();
    for (const t of targets) {
      keep.add(t.key);
      let it = items.get(t.key);
      if (!it) {
        it = { key: t.key, x: null, y: null, alpha: 0, dying: false };
        items.set(t.key, it);
      }
      it.dying = false;
      it.track = t.track;
      it.color = t.color;
      it.label = t.label;
      it.kind = t.kind;
      it.t0 = resultTime;
      if (t.track.outline) it.outline = t.track.outline;
    }
    for (const it of items.values()) if (!keep.has(it.key)) it.dying = true;
  }

  function clearAll() { items.clear(); }

  function render(now, dt, view, map, extra) {
    const { w, h, dpr } = view;
    if (!items.size && !extra?.target && !drewLast) return 0;
    const t0 = performance.now();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    drewLast = false;
    if (extra?.target) { drawTarget(extra.target, now); drewLast = true; }
    const ease = 1 - Math.exp(-dt / 55);
    for (const it of [...items.values()]) {
      // Fade in quickly; short fade-out once a lock is actually dropped.
      it.alpha += (it.dying ? -1 : 1) * dt / (it.dying ? 280 : 160);
      if (it.alpha <= 0 && it.dying) { items.delete(it.key); continue; }
      it.alpha = Math.min(1, Math.max(0, it.alpha));
      const tr = it.track;
      // Predict where the sock is now from its velocity (capped).
      const lead = Math.min(140, Math.max(0, now - it.t0));
      const px = tr.cx + (tr.vx || 0) * lead, py = tr.cy + (tr.vy || 0) * lead;
      const [tx, ty] = map.toView(px, py);
      if (it.x === null) { it.x = tx; it.y = ty; } else { it.x += (tx - it.x) * ease; it.y += (ty - it.y) * ease; }
      drawItem(it, map.scale, map.mirror, now);
      drewLast = true;
    }
    return performance.now() - t0;
  }

  function drawItem(it, scale, mirror, now) {
    const o = it.outline;
    if (!o) return;
    ctx.save();
    ctx.globalAlpha = it.alpha;
    ctx.beginPath();
    const sx = mirror ? -scale : scale;
    let topY = Infinity, topX = it.x;
    for (let k = 0; k < o.length; k += 2) {
      const x = it.x + o[k] * sx, y = it.y + o[k + 1] * scale;
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      if (y < topY) { topY = y; topX = x; }
    }
    ctx.closePath();
    const c = it.color;
    if (it.kind === 'ref') {
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.stroke();
      ctx.restore();
      pill(it.x, topY - 12, 'scanned', 'rgba(255,255,255,0.92)', '#111', it.alpha);
      return;
    }
    const pulse = 0.5 + 0.5 * Math.sin(now * 0.005);
    ctx.fillStyle = rgba(c, 0.16 + 0.06 * pulse);
    ctx.fill();
    // Glow without shadowBlur (cheap on Safari): wide translucent strokes.
    ctx.lineJoin = 'round';
    ctx.strokeStyle = rgba(c, 0.18);
    ctx.lineWidth = 14;
    ctx.stroke();
    ctx.strokeStyle = rgba(c, 0.35 + 0.15 * pulse);
    ctx.lineWidth = 7;
    ctx.stroke();
    ctx.strokeStyle = c;
    ctx.lineWidth = it.kind === 'color' ? 2.5 : 3;
    ctx.stroke();
    ctx.restore();
    if (it.label) {
      if (it.kind === 'pair') badge(it.x, it.y, it.label, c, it.alpha);
      else if (it.kind !== 'color') pill(it.x, Math.max(14, topY - 14), it.label, c, '#06240f', it.alpha);
    }
  }

  function drawTarget(t, now) {
    const { x, y, w, h } = t;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 3;
    const L = Math.min(w, h) * 0.22;
    ctx.beginPath();
    // Corner brackets.
    ctx.moveTo(x, y + L); ctx.lineTo(x, y); ctx.lineTo(x + L, y);
    ctx.moveTo(x + w - L, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + L);
    ctx.moveTo(x + w, y + h - L); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - L, y + h);
    ctx.moveTo(x + L, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - L);
    ctx.stroke();
    ctx.restore();
    pill(x + w / 2, y + h + 18, 'Put one sock here, tap Scan', 'rgba(20,20,24,0.8)', '#fff', 1);
  }

  function badge(x, y, text, color, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(x, y, 15, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = '800 15px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, y + 0.5);
    ctx.restore();
  }

  function pill(x, y, text, bg, fg, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = '800 13px -apple-system, BlinkMacSystemFont, sans-serif';
    const tw = ctx.measureText(text).width;
    const pw = tw + 16, ph = 22;
    const rx = x - pw / 2, ry = y - ph / 2;
    ctx.beginPath();
    ctx.moveTo(rx + 11, ry);
    ctx.arcTo(rx + pw, ry, rx + pw, ry + ph, 11);
    ctx.arcTo(rx + pw, ry + ph, rx, ry + ph, 11);
    ctx.arcTo(rx, ry + ph, rx, ry, 11);
    ctx.arcTo(rx, ry, rx + pw, ry, 11);
    ctx.closePath();
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.fillStyle = fg;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, y + 0.5);
    ctx.restore();
  }

  return { setTargets, render, clearAll, count: () => items.size };
}

const rgbCache = new Map();
function rgba(hex, a) {
  let rgb = rgbCache.get(hex);
  if (!rgb) {
    const n = parseInt(hex.slice(1), 16);
    rgb = `${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}`;
    rgbCache.set(hex, rgb);
  }
  return `rgba(${rgb}, ${a.toFixed(3)})`;
}

export { MATCH_COLOR };
