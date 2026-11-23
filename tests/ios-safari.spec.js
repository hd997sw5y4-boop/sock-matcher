import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const FRAME = path.resolve('/workspace/sock-matcher-feedback/v1257/f01.jpg');

test.beforeEach(({ browserName }) => {
  test.skip(browserName !== 'webkit', 'WebKit / iPhone Safari only');
});

test.use({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 3,
  isMobile: true,
  hasTouch: true,
  userAgent:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
});

function collectErrors(page) {
  const errors = [];
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });
  page.on('pageerror', (err) => errors.push(String(err)));
  return errors;
}

async function blockWebGPU(page) {
  await page.addInitScript(() => {
    try { Object.defineProperty(navigator, 'gpu', { get: () => undefined, configurable: true }); }
    catch { /* ignore */ }
  });
}

test('battery saver and a tight power budget are the default', async ({ page }) => {
  await blockWebGPU(page);
  await page.goto('/');
  const power = await page.evaluate(() => window.__sockMatcher.getState());
  expect(power.batterySaver).toBe(true);
  expect(power.power.maxSide).toBe(320);
  expect(power.power.classicHz).toBeLessThanOrEqual(8.1);
  expect(power.power.classicHz).toBeGreaterThanOrEqual(7);
  expect(power.power.mlInterval).toBeGreaterThanOrEqual(600);
  expect(power.power.maxCrops).toBeLessThanOrEqual(6);
  await expect(page.locator('#battery-saver')).toBeChecked();
});

test('classic scan on the demo pile works immediately, even with models enabled', async ({ page }) => {
  const errors = collectErrors(page);
  await blockWebGPU(page);
  await page.goto('/');
  await page.locator('#btn-demo-gate').click();
  await expect(page.locator('#gate')).toBeHidden();
  await page.waitForFunction(() => {
    const s = window.__sockMatcher?.getState();
    return s && s.source === 'demo' && s.tracks.length >= 8;
  }, null, { timeout: 15_000 });

  const before = await page.evaluate(() => window.__sockMatcher.getState());
  expect(before.hasRef).toBeFalsy();
  expect(before.ml.status === 'idle' || before.ml.status === 'loading').toBeTruthy();

  const first = before.tracks[0];
  await page.evaluate(async ({ x, y }) => {
    await window.__sockMatcher.tapWork(x, y);
  }, { x: first.cx, y: first.cy });

  await page.waitForFunction(() => window.__sockMatcher.getState().hasRef, null, { timeout: 8_000 });
  const chip = await page.locator('#status-chip').textContent();
  expect(chip).not.toMatch(/Still loading the pattern model/i);
  expect(chip).not.toMatch(/Picking the sock/i);
  await expect(page.locator('#ref-card')).toBeVisible();
  const after = await page.evaluate(() => window.__sockMatcher.getState());
  expect(after.hasRef).toBe(true);
  expect((after.displayed?.matches || after.matches || []).length).toBeLessThanOrEqual(3);
  // Classic path must not have been blocked by a failed or unfinished model.
  expect(errors.filter((e) => /Still loading|Picking the sock/i.test(e))).toEqual([]);
  await page.waitForFunction(() => window.__sockMatcher.getState().ml.status === 'ready', null, { timeout: 90_000 });
  const ml = await page.evaluate(() => window.__sockMatcher.getState().ml);
  expect(ml.status).toBe('ready');
  expect(ml.dino).toBe(true);
});

test('a private v1257 frame can be uploaded and scanned without waiting on models', async ({ page }) => {
  test.skip(!fs.existsSync(FRAME), 'private v1257 frames are not on this machine');
  await blockWebGPU(page);
  const remote = [];
  page.on('request', (req) => {
    const url = req.url();
    if (/huggingface\.co|cdn\.jsdelivr\.net/i.test(url)) remote.push(url);
  });
  await page.goto('/');
  await page.locator('#file-input').setInputFiles(FRAME);
  await expect(page.locator('#gate')).toBeHidden({ timeout: 15_000 });
  await page.waitForFunction(() => window.__sockMatcher?.getState()?.source === 'photo', null, { timeout: 15_000 });

  await page.locator('#btn-shutter').click();
  // The shutter is Scan. Classic work must be queued immediately — no blocking toast.
  await page.waitForTimeout(800);
  const chip = (await page.locator('#status-chip').textContent()) || '';
  expect(chip).not.toMatch(/Still loading the pattern model/i);
  expect(chip).not.toMatch(/Picking the sock/i);
  const state = await page.evaluate(() => window.__sockMatcher.getState());
  expect(state.source).toBe('photo');
  expect(['idle', 'loading', 'ready', 'error']).toContain(state.ml.status);
  expect(remote, `models must be same-origin, got ${remote.join(', ')}`).toEqual([]);
});

test('model and WASM assets are served same-origin', async ({ request, baseURL }) => {
  const paths = [
    '/models/Xenova/dinov2-small/config.json',
    '/models/Xenova/dinov2-small/preprocessor_config.json',
    '/models/Xenova/dinov2-small/onnx/model_q4.onnx',
    '/vendor/transformers.web.js',
    '/vendor/onnx/ort-wasm-simd-threaded.wasm',
    '/vendor/onnx/ort-wasm-simd-threaded.mjs',
  ];
  for (const p of paths) {
    const res = await request.get(`${baseURL}${p}`);
    expect(res.ok(), `${p} -> ${res.status()}`).toBe(true);
    expect(Number(res.headers()['content-length'] || 0) || (await res.body()).length).toBeGreaterThan(100);
  }
});
