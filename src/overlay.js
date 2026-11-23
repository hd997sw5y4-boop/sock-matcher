export const PAIR_COLORS = [
  '#FF3B30', '#0A84FF', '#30D158', '#FFD60A', '#BF5AF2',
  '#FF9F0A', '#64D2FF', '#FF375F', '#32ADE6', '#00C7BE',
];

export function colorForPair(num) {
  return PAIR_COLORS[(num - 1) % PAIR_COLORS.length];
}

export function createMapper(srcW, srcH, viewW, viewH, fit = 'cover') {
  const scale = (fit === 'contain' ? Math.min : Math.max)(viewW / srcW, viewH / srcH);
  const dw = srcW * scale, dh = srcH * scale;
  const ox = (viewW - dw) / 2, oy = (viewH - dh) / 2;
  return {
    scale, ox, oy,
    srcToView(x, y) { return [x * scale + ox, y * scale + oy]; },
    viewToSrc(x, y) { return [(x - ox) / scale, (y - oy) / scale]; },
    workToView(x, y, workW, workH) {
      return this.srcToView(x * (srcW / workW), y * (srcH / workH));
    },
    viewToWork(x, y, workW, workH) {
      const [sx, sy] = this.viewToSrc(x, y);
      return [sx * (workW / srcW), sy * (workH / srcH)];
    },
  };
}

export function drawDisplayFrame(ctx, source, viewW, viewH, srcW, srcH, mirror = false, fit = 'cover') {
  ctx.save();
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, viewW, viewH);
  const scale = (fit === 'contain' ? Math.min : Math.max)(viewW / srcW, viewH / srcH);
  const dw = srcW * scale, dh = srcH * scale;
  const ox = (viewW - dw) / 2, oy = (viewH - dh) / 2;
  if (mirror) {
    ctx.translate(viewW, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(source, viewW - (ox + dw), oy, dw, dh);
  } else {
    ctx.drawImage(source, ox, oy, dw, dh);
  }
  ctx.restore();
}

export function renderOverlay(ctx, state, t) {
  const { viewW, viewH, dpr, mapper, workW, workH, mode, socks, pairs, unmatched, selected, matches, frozen } = state;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, viewW, viewH);

  if (!socks || !socks.length) return;

  const toView = (pt) => mapper.workToView(pt[0], pt[1], workW, workH);

  if (mode === 'tap' && selected) {
    // Dim the whole frame, punch socks later by drawing them on top of a veil.
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(0, 0, viewW, viewH);
  }

  const pairOf = new Map();
  for (const p of pairs) {
    pairOf.set(p.a, p);
    pairOf.set(p.b, p);
  }

  const pulse = 0.5 + 0.5 * Math.sin(t * 0.006);

  for (const sock of socks) {
    const pair = pairOf.get(sock.index);
    const isSel = selected && selected.index === sock.index;
    const matchInfo = matches?.find((m) => m.index === sock.index);
    const isMatch = !!matchInfo;
    const dim = mode === 'tap' && selected && !isSel && !isMatch;

    let color = '#d4d4d8';
    let width = 2.4;
    let dashed = false;
    let fillA = 0.14;

    if (mode === 'pairs') {
      if (pair) {
        color = colorForPair(pair.num);
        width = 3;
        fillA = 0.18;
      } else {
        color = 'rgba(255,255,255,0.7)';
        dashed = true;
        width = 2;
        fillA = 0.06;
      }
    } else if (mode === 'tap') {
      if (isSel) {
        color = '#ffffff';
        width = 4;
        fillA = 0.16;
      } else if (isMatch) {
        color = '#30D158';
        width = 3.5 + pulse * 1.5;
        fillA = 0.22 + pulse * 0.12;
      } else {
        color = 'rgba(255,255,255,0.28)';
        width = 1.5;
        fillA = 0.04;
      }
    }

    if (dim) fillA *= 0.3;

    const contour = sock.contour?.map(toView) || boxContour(sock, toView);
    drawContour(ctx, contour, color, width, fillA, dashed);
  }

  // Badges
  if (mode === 'pairs') {
    for (const p of pairs) {
      const color = colorForPair(p.num);
      const sa = socks[p.a], sb = socks[p.b];
      if (sa) drawBadge(ctx, ...toView([sa.cx, sa.cy]), String(p.num), color);
      if (sb) drawBadge(ctx, ...toView([sb.cx, sb.cy]), String(p.num), color);
    }
    for (const s of unmatched) {
      const [x, y] = toView([s.cx, s.minY - 4]);
      drawTiny(ctx, x, y, 'solo', 'rgba(24,24,28,0.82)', '#fff');
    }
  } else if (mode === 'tap' && selected) {
    const [sx, sy] = toView([selected.cx, selected.minY - 6]);
    drawTiny(ctx, sx, sy, 'this sock', '#fff', '#111');
    if (!matches?.length) {
      const [x, y] = toView([selected.cx, selected.maxY + 16]);
      drawTiny(ctx, x, y, 'no clear match', 'rgba(24,24,28,0.9)', '#fff');
    }
    for (const m of matches || []) {
      const sock = socks[m.index];
      if (!sock) continue;
      const [x, y] = toView([sock.cx, sock.minY - 6]);
      const label = `${Math.round(m.score * 100)}%`;
      drawBadge(ctx, x, y, label, '#30D158', 22);
    }
  }

  if (frozen) {
    ctx.fillStyle = 'rgba(10,132,255,0.12)';
    ctx.fillRect(0, 0, viewW, 0); // no-op reserved
  }
}

function boxContour(sock, toView) {
  return [
    toView([sock.minX, sock.minY]),
    toView([sock.maxX, sock.minY]),
    toView([sock.maxX, sock.maxY]),
    toView([sock.minX, sock.maxY]),
  ];
}

function drawContour(ctx, pts, color, width, fillA, dashed) {
  if (!pts || pts.length < 3) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.fillStyle = hexAlpha(color, fillA);
  ctx.fill();
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.setLineDash(dashed ? [6, 5] : []);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawBadge(ctx, x, y, text, color, r = 15) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(0,0,0,0.25)';
  ctx.stroke();
  ctx.fillStyle = contrastInk(color);
  ctx.font = `700 ${text.length > 2 ? 11 : 14}px -apple-system, BlinkMacSystemFont, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y + 0.5);
}

function drawTiny(ctx, x, y, text, bg, fg) {
  ctx.font = '700 11px -apple-system, BlinkMacSystemFont, sans-serif';
  const tw = ctx.measureText(text).width;
  const pad = 7;
  const h = 18;
  const w = tw + pad * 2;
  roundRect(ctx, x - w / 2, y - h / 2, w, h, 9);
  ctx.fillStyle = bg;
  ctx.fill();
  ctx.fillStyle = fg;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y + 0.5);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function hexAlpha(color, a) {
  if (color.startsWith('rgba') || color.startsWith('rgb')) {
    return color.replace(/rgba?\(([^)]+)\)/, (_, inner) => {
      const parts = inner.split(',').map((s) => s.trim());
      return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${a})`;
    });
  }
  const h = color.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function contrastInk(color) {
  if (!color.startsWith('#')) return '#111';
  const h = color.slice(1);
  const n = parseInt(h, 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return lum > 170 ? '#111' : '#fff';
}

export function hitTestSock(socks, workX, workY) {
  let best = null, bestD = Infinity;
  for (const s of socks) {
    const pad = 8;
    if (workX < s.minX - pad || workX > s.maxX + pad || workY < s.minY - pad || workY > s.maxY + pad) continue;
    const d = Math.hypot(workX - s.cx, workY - s.cy);
    if (d < bestD) { bestD = d; best = s; }
  }
  return best;
}
