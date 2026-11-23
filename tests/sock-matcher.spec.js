import { test, expect } from '@playwright/test';

function collectErrors(page) {
  const errors = [];
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });
  page.on('pageerror', (err) => errors.push(String(err)));
  return errors;
}

async function openDemo(page) {
  await page.goto('/?demo=1');
  await expect(page.locator('#gate')).toBeHidden();
  await page.waitForFunction(() => {
    const s = window.__sockMatcher?.getState();
    return s && s.source === 'demo' && s.tracks.length >= 10;
  }, null, { timeout: 15_000 });
}

/** Map every track to its nearest ground-truth demo sock. */
const TRUTH_FN = `(s) => {
  const toSrc = (t) => [s.rect.sx + t.cx * s.rect.sw / s.work.w, s.rect.sy + t.cy * s.rect.sh / s.work.h];
  const truthOf = new Map();
  for (const t of s.tracks) {
    const [x, y] = toSrc(t);
    let best = null, bd = Infinity;
    for (const d of s.demoTruth) { const dd = Math.hypot(d.x - x, d.y - y); if (dd < bd) { bd = dd; best = d; } }
    if (bd < 110) truthOf.set(t.id, best);
  }
  return truthOf;
}`;

test('demo pile: engine finds the real pairs', async ({ page }) => {
  const errors = collectErrors(page);
  await openDemo(page);
  const res = await page.evaluate(async (fn) => {
    const truth = eval(fn);
    const s = window.__sockMatcher.getState();
    const t = truth(s);
    const all = await window.__sockMatcher.allPairs();
    const ok = all.filter((p) => t.get(p.a) && t.get(p.b) && t.get(p.a).pairId === t.get(p.b).pairId && t.get(p.a) !== t.get(p.b));
    return { tracks: s.tracks.length, total: all.length, correct: ok.length };
  }, TRUTH_FN);
  expect(res.tracks).toBeGreaterThanOrEqual(14);
  expect(res.correct, JSON.stringify(res)).toBeGreaterThanOrEqual(6);
  expect(res.correct / Math.max(1, res.total)).toBeGreaterThanOrEqual(0.85);
  expect(errors, errors.join('\n')).toEqual([]);
});

test('Scan a Sock is the default and tapping a sock lights up its mate', async ({ page }) => {
  const errors = collectErrors(page);
  await openDemo(page);
  await expect(page.locator('#mode-scan')).toHaveClass(/active/);
  const out = await page.evaluate(async (fn) => {
    const truth = eval(fn);
    const api = window.__sockMatcher;
    const s0 = api.getState();
    const t = truth(s0);
    const results = [];
    for (const tr of s0.tracks) {
      const me = t.get(tr.id);
      if (!me) continue;
      const s = await api.tapWork(tr.cx, tr.cy);
      const top = s.matches[0];
      const topTruth = top && truth(s).get(top.id);
      results.push({
        style: me.style, n: s.matches.length,
        ok: !!topTruth && topTruth.pairId === me.pairId && topTruth !== me,
        score: top ? top.score : 0, overlay: s.overlayItems,
      });
    }
    return results;
  }, TRUTH_FN);
  const correct = out.filter((r) => r.ok).length;
  expect(out.length).toBeGreaterThanOrEqual(14);
  expect(correct, JSON.stringify(out, null, 1)).toBeGreaterThanOrEqual(Math.ceil(out.length * 0.8));
  for (const r of out) expect(r.n).toBeLessThanOrEqual(3);
  await expect(page.locator('#ref-card')).toBeVisible();
  expect(errors, errors.join('\n')).toEqual([]);
});

test('Auto Pairs shows at most 3 confident, correct pairs and no solo badges', async ({ page }) => {
  const errors = collectErrors(page);
  await openDemo(page);
  const res = await page.evaluate(async (fn) => {
    const truth = eval(fn);
    const api = window.__sockMatcher;
    const p = api.nextResult();
    api.setMode('pairs');
    await p;
    await new Promise((r) => setTimeout(r, 400));
    const s = api.getState();
    const t = truth(s);
    const c = document.getElementById('overlay');
    const { data } = c.getContext('2d').getImageData(0, 0, c.width, c.height);
    let painted = 0;
    for (let i = 3; i < data.length; i += 64) if (data[i] > 20) painted++;
    return {
      pairs: s.pairs.length,
      correct: s.pairs.filter((q) => t.get(q.a) && t.get(q.b) && t.get(q.a).pairId === t.get(q.b).pairId).length,
      painted,
    };
  }, TRUTH_FN);
  expect(res.pairs).toBeGreaterThanOrEqual(1);
  expect(res.pairs).toBeLessThanOrEqual(3);
  expect(res.correct).toBe(res.pairs);
  expect(res.painted).toBeGreaterThan(50);
  const html = await page.content();
  expect(html.toLowerCase()).not.toContain('solo');
  expect(errors, errors.join('\n')).toEqual([]);
});

test('gate screen offers camera, demo, and upload', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#gate')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Enable Camera' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Try Demo Pile' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Upload Photo' })).toBeVisible();
  await page.getByRole('button', { name: 'Try Demo Pile' }).click();
  await expect(page.locator('#gate')).toBeHidden();
  await expect(page.locator('#status-chip')).toBeVisible();
});
