import { test, expect } from '@playwright/test';

test('demo pile auto-pairs and tap-to-match work without console errors', async ({ page }) => {
  const errors = [];
  const pageErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', (err) => pageErrors.push(String(err)));

  await page.goto('/?demo=1');
  await expect(page.locator('#gate')).toBeHidden();

  const state = await page.waitForFunction(() => {
    const api = window.__sockMatcher;
    if (!api) return null;
    const s = api.getState();
    if (s.source === 'demo' && s.socks.length >= 10 && s.pairs.length >= 6) return s;
    return null;
  }, { timeout: 15_000 });

  const snapshot = await page.evaluate(() => window.__sockMatcher.getState());
  expect(snapshot.source).toBe('demo');
  expect(snapshot.mode).toBe('pairs');
  expect(snapshot.socks.length).toBeGreaterThanOrEqual(14);
  expect(snapshot.pairs.length).toBeGreaterThanOrEqual(6);

  // Map each detection to the nearest ground-truth sock, then check pair IDs agree.
  const accuracy = await page.evaluate(() => {
    const s = window.__sockMatcher.getState();
    const { src, work, demoTruth, pairs, socks } = s;
    const toSrc = (wx, wy) => [wx * (src.w / work.w), wy * (src.h / work.h)];
    const nearest = (wx, wy) => {
      const [x, y] = toSrc(wx, wy);
      let best = null, bestD = Infinity;
      for (const t of demoTruth) {
        const d = Math.hypot(t.x - x, t.y - y);
        if (d < bestD) { bestD = d; best = t; }
      }
      return { truth: best, dist: bestD };
    };
    const results = pairs.map((p) => {
      const a = socks[p.a], b = socks[p.b];
      const na = nearest(a.cx, a.cy);
      const nb = nearest(b.cx, b.cy);
      return {
        num: p.num,
        score: p.score,
        ok: na.truth && nb.truth && na.truth.pairId === nb.truth.pairId && na.dist < 120 && nb.dist < 120,
        pairA: na.truth?.pairId,
        pairB: nb.truth?.pairId,
        styleA: na.truth?.style,
        styleB: nb.truth?.style,
        distA: na.dist,
        distB: nb.dist,
      };
    });
    const correct = results.filter((r) => r.ok).length;
    return { correct, total: results.length, socks: socks.length, results };
  });

  expect(accuracy.socks, `detections=${accuracy.socks}`).toBeGreaterThanOrEqual(14);
  expect(accuracy.correct, JSON.stringify(accuracy.results, null, 2)).toBeGreaterThanOrEqual(6);
  expect(accuracy.correct / accuracy.total).toBeGreaterThanOrEqual(0.75);

  // Overlays should have painted pair badges (non-blank overlay canvas).
  const overlayPainted = await page.evaluate(() => {
    const c = document.getElementById('overlay');
    const ctx = c.getContext('2d');
    const { data } = ctx.getImageData(0, 0, c.width, c.height);
    let colored = 0;
    for (let i = 0; i < data.length; i += 16) {
      if (data[i + 3] > 20) colored++;
    }
    return colored;
  });
  expect(overlayPainted).toBeGreaterThan(50);

  // Tap-a-Sock: pick a known paired sock and confirm its mate lights up.
  await page.evaluate(() => window.__sockMatcher.setMode('tap'));
  const tap = await page.evaluate(() => {
    const api = window.__sockMatcher;
    const s = api.getState();
    const p = s.pairs[0];
    const sock = s.socks[p.a];
    const after = api.tapWork(sock.cx, sock.cy);
    return {
      selected: after.selected,
      expectedMate: p.b,
      matches: after.matches,
      pairScore: p.score,
    };
  });

  expect(tap.selected).toBeGreaterThanOrEqual(0);
  expect(tap.matches.length).toBeGreaterThanOrEqual(1);
  const mateHit = tap.matches.some((m) => m.index === tap.expectedMate);
  expect(mateHit, JSON.stringify(tap)).toBeTruthy();
  expect(tap.matches[0].score).toBeGreaterThan(0.5);

  expect(pageErrors, pageErrors.join('\n')).toEqual([]);
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
