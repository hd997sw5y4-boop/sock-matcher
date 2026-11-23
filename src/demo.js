/**
 * Deterministic, realistic-looking laundry pile used for instant testing
 * and Playwright verification. Each sock has a known pairId.
 */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const STYLES = [
  { id: 'stripe-red', kind: 'stripe', body: '#c81e1e', stripe: '#f4f4f5', cuff: '#991b1b', toe: '#7f1d1d' },
  { id: 'navy-solid', kind: 'heather', body: '#1e3a8a', cuff: '#172554', toe: '#172554' },
  { id: 'dot-teal', kind: 'dots', body: '#0f766e', dot: '#ecfdf5', cuff: '#115e59', toe: '#134e4a' },
  { id: 'mustard-stripe', kind: 'thin-stripe', body: '#ca8a04', stripe: '#422006', cuff: '#a16207', toe: '#854d0e' },
  { id: 'purple-block', kind: 'colorblock', body: '#6d28d9', cuff: '#f9a8d4', toe: '#f472b6' },
  { id: 'forest-argyle', kind: 'argyle', body: '#166534', accent: '#dcfce7', line: '#052e16', cuff: '#14532d', toe: '#14532d' },
  { id: 'coral-solid', kind: 'heather', body: '#f97316', cuff: '#c2410c', toe: '#9a3412' },
  { id: 'sky-stripe', kind: 'stripe', body: '#38bdf8', stripe: '#f0f9ff', cuff: '#0284c7', toe: '#0369a1' },
];

export const DEMO_SIZE = { w: 1080, h: 1920 };

export function generateDemoPile(seed = 20261123) {
  const rand = mulberry32(seed);
  const w = DEMO_SIZE.w, h = DEMO_SIZE.h;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  paintLinen(ctx, w, h, rand);

  const socks = [];
  // Two of each style, scattered so mates are not always neighbors.
  const jobs = [];
  STYLES.forEach((style, pairId) => {
    jobs.push({ style, pairId, copy: 0 });
    jobs.push({ style, pairId, copy: 1 });
  });
  // Shuffle job order (deterministic).
  for (let i = jobs.length - 1; i > 0; i--) {
    const j = (rand() * (i + 1)) | 0;
    [jobs[i], jobs[j]] = [jobs[j], jobs[i]];
  }

  // Jittered 4×4 grid so socks stay clearly separated for both eyes and the detector.
  const cols = 4, rows = 4;
  const cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      cells.push({
        x: 150 + c * ((w - 300) / (cols - 1)) + (rand() - 0.5) * 44,
        y: 340 + r * ((h - 820) / (rows - 1)) + (rand() - 0.5) * 36,
      });
    }
  }
  const placed = [];
  for (let i = 0; i < jobs.length; i++) {
    const job = jobs[i];
    const cell = cells[i];
    const pose = {
      x: cell.x,
      y: cell.y,
      rot: (rand() - 0.5) * Math.PI * 0.9,
      length: 200 + rand() * 28,
      width: 76 + rand() * 12,
    };
    placed.push(pose);
    drawSock(ctx, pose, job.style, rand, job.copy);
    socks.push({
      pairId: job.pairId,
      style: job.style.id,
      copy: job.copy,
      x: pose.x,
      y: pose.y,
      rot: pose.rot,
      length: pose.length,
      width: pose.width,
    });
  }

  return { canvas, socks, seed, width: w, height: h };
}

function sep(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function paintLinen(ctx, w, h, rand) {
  ctx.fillStyle = '#cfc4b3';
  ctx.fillRect(0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const weave = ((x * 13 + y * 7) % 9 === 0 ? -7 : 0) + ((x + y * 3) % 11 === 0 ? 6 : 0);
      const wrinkle = Math.sin(x * 0.011 + y * 0.017) * 8 + Math.sin(x * 0.03 - y * 0.01) * 5;
      const n = (rand() - 0.5) * 10;
      const v = weave + wrinkle + n;
      d[i] = clamp(207 + v);
      d[i + 1] = clamp(196 + v * 0.9);
      d[i + 2] = clamp(179 + v * 0.75);
      d[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  // Soft vignette so the pile still reads as a photo.
  const g = ctx.createRadialGradient(w * 0.5, h * 0.48, w * 0.2, w * 0.5, h * 0.5, w * 0.75);
  g.addColorStop(0, 'rgba(0,0,0,0)');
  g.addColorStop(1, 'rgba(80,60,40,0.16)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
}

function clamp(v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

function sockPath(ctx, length, width) {
  const hw = width / 2;
  const cuff = length * 0.18;
  const heel = length * 0.18;
  const mid = length * 0.42;
  ctx.beginPath();
  ctx.moveTo(-hw * 0.92, -length / 2);
  ctx.lineTo(hw * 0.92, -length / 2);
  ctx.quadraticCurveTo(hw * 1.02, -length / 2 + cuff * 0.4, hw, -length / 2 + cuff);
  ctx.quadraticCurveTo(hw * 1.05, 0, hw * 0.95, mid - length / 2);
  // heel bulge on the right
  ctx.quadraticCurveTo(hw * 1.35, mid - length / 2 + heel * 0.4, hw * 0.85, mid - length / 2 + heel);
  ctx.quadraticCurveTo(hw * 0.95, length / 2 - 10, 0, length / 2);
  ctx.quadraticCurveTo(-hw * 0.95, length / 2 - 10, -hw * 0.72, mid - length / 2 + heel);
  ctx.quadraticCurveTo(-hw * 1.0, mid - length / 2, -hw, -length / 2 + cuff);
  ctx.quadraticCurveTo(-hw * 1.02, -length / 2 + cuff * 0.4, -hw * 0.92, -length / 2);
  ctx.closePath();
}

function drawSock(ctx, pose, style, rand, copy) {
  const { x, y, rot, length, width } = pose;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rot);
  if (copy) ctx.scale(-1, 1);
  // Slight scale jitter so mates aren't pixel-identical.
  const s = 0.96 + rand() * 0.08;
  ctx.scale(s, s);

  // Soft contact shadow.
  ctx.save();
  ctx.translate(8, 10);
  sockPath(ctx, length, width);
  ctx.fillStyle = 'rgba(40, 30, 20, 0.22)';
  ctx.filter = 'blur(8px)';
  ctx.fill();
  ctx.restore();
  ctx.filter = 'none';

  sockPath(ctx, length, width);
  ctx.clip();

  ctx.fillStyle = style.body;
  ctx.fillRect(-width, -length, width * 2, length * 2);

  paintPattern(ctx, style, length, width, rand);

  // Cuff ribbing
  const cuffH = length * 0.16;
  ctx.fillStyle = style.cuff || style.body;
  ctx.fillRect(-width, -length / 2, width * 2, cuffH);
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 2;
  for (let i = 0; i < 6; i++) {
    const yy = -length / 2 + 4 + i * (cuffH / 6);
    ctx.beginPath();
    ctx.moveTo(-width, yy);
    ctx.lineTo(width, yy);
    ctx.stroke();
  }

  // Toe cap
  ctx.fillStyle = style.toe || style.body;
  ctx.beginPath();
  ctx.ellipse(0, length / 2 - 16, width * 0.48, 22, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();

  // Second pass: highlight / shade without leaving clip (re-apply transform).
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rot);
  if (copy) ctx.scale(-1, 1);
  ctx.scale(s, s);
  sockPath(ctx, length, width);
  ctx.clip();
  const lg = ctx.createLinearGradient(-width, 0, width, 0);
  lg.addColorStop(0, 'rgba(0,0,0,0.18)');
  lg.addColorStop(0.45, 'rgba(255,255,255,0.08)');
  lg.addColorStop(1, 'rgba(0,0,0,0.14)');
  ctx.fillStyle = lg;
  ctx.fillRect(-width, -length, width * 2, length * 2);

  // Outline
  sockPath(ctx, length, width);
  ctx.strokeStyle = 'rgba(30,20,15,0.35)';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function paintPattern(ctx, style, length, width, rand) {
  if (style.kind === 'stripe') {
    ctx.strokeStyle = style.stripe;
    ctx.lineWidth = 13;
    for (let y = -length / 2 + 28; y < length / 2; y += 26) {
      ctx.beginPath();
      ctx.moveTo(-width, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  } else if (style.kind === 'thin-stripe') {
    ctx.strokeStyle = style.stripe;
    ctx.lineWidth = 4;
    for (let y = -length / 2 + 22; y < length / 2; y += 12) {
      ctx.beginPath();
      ctx.moveTo(-width, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  } else if (style.kind === 'dots') {
    ctx.fillStyle = style.dot;
    for (let y = -length / 2 + 30; y < length / 2 - 10; y += 22) {
      const odd = ((y / 22) | 0) % 2;
      for (let x = -width * 0.6 + (odd ? 10 : 0); x < width * 0.6; x += 20) {
        ctx.beginPath();
        ctx.arc(x, y, 4.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  } else if (style.kind === 'argyle') {
    const step = 36;
    ctx.strokeStyle = style.line;
    ctx.fillStyle = style.accent;
    ctx.lineWidth = 1.5;
    for (let y = -length / 2 + 30; y < length / 2; y += step) {
      for (let x = -width * 0.7; x < width * 0.7; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, y - 14);
        ctx.lineTo(x + 12, y);
        ctx.lineTo(x, y + 14);
        ctx.lineTo(x - 12, y);
        ctx.closePath();
        if (((x + y) / step | 0) % 2 === 0) ctx.fill();
        ctx.stroke();
      }
    }
  } else if (style.kind === 'colorblock') {
    ctx.fillStyle = style.cuff;
    ctx.fillRect(-width, -length / 2, width * 2, length * 0.22);
    ctx.fillStyle = style.toe;
    ctx.fillRect(-width, length / 2 - 48, width * 2, 80);
  } else if (style.kind === 'heather') {
    ctx.fillStyle = 'rgba(255,255,255,0.07)';
    for (let i = 0; i < 180; i++) {
      ctx.fillRect((rand() - 0.5) * width * 1.6, (rand() - 0.5) * length, 2, 2);
    }
    ctx.fillStyle = 'rgba(0,0,0,0.08)';
    for (let i = 0; i < 140; i++) {
      ctx.fillRect((rand() - 0.5) * width * 1.6, (rand() - 0.5) * length, 2, 2);
    }
  }
}

/** Map a sock pose to an axis-aligned box in image pixels (loose). */
export function sockBounds(sock) {
  const r = Math.hypot(sock.length, sock.width) * 0.55;
  return { x: sock.x - r, y: sock.y - r, w: r * 2, h: r * 2, cx: sock.x, cy: sock.y };
}
