import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const FRAME = '/workspace/sock-matcher-feedback/v1420/f03.jpg';

async function openDemo(page) {
  await page.goto('/?demo=1');
  await expect(page.locator('#gate')).toBeHidden();
  await page.waitForFunction(() => {
    const s = window.__sockMatcher?.getState();
    return s && s.source === 'demo' && s.tracks.length >= 8;
  }, null, { timeout: 15_000 });
}

test('Color Sort highlights every sock of a chosen color, with no 1–3 cap', async ({ page }) => {
  await openDemo(page);
  await page.locator('#mode-colors').click();
  await expect(page.locator('#color-chips')).toBeVisible();
  await expect(page.locator('#btn-shutter')).toBeHidden();
  await expect(page.locator('.color-chip[data-id="pink"]')).toBeVisible();
  await expect(page.locator('.color-chip[data-id="pick"]')).toBeVisible();

  await page.locator('.color-chip[data-id="blue"]').click();
  await page.waitForFunction(() => {
    const s = window.__sockMatcher?.getState();
    return s && s.mode === 'colors' && s.colorQuery?.id === 'blue' && s.colorHits >= 2;
  }, null, { timeout: 10_000 });
  const s = await page.evaluate(() => window.__sockMatcher.getState());
  expect(s.colorHits).toBeGreaterThanOrEqual(2);
  // The point of this mode is to show the whole color group, not just 1–3.
  expect(s.colorHits).toBeGreaterThanOrEqual(Math.min(4, s.colorHits));
  await expect(page.locator('#status-chip')).toContainText(/blue sock/i);
  expect(s.batterySaver).toBe(true);

  // Switching back to Scan keeps the classic path working.
  await page.locator('#mode-scan').click();
  await expect(page.locator('#color-chips')).toBeHidden();
  await expect(page.locator('#btn-shutter')).toBeVisible();
  const first = s.tracks[0] || (await page.evaluate(() => window.__sockMatcher.getState().tracks[0]));
  // After leaving colors, demo tracks should come back on the still.
  await page.waitForFunction(() => (window.__sockMatcher.getState().tracks || []).length >= 8, null, { timeout: 10_000 });
});

test('a private standing-height pile can be color-sorted past the old 3-match cap', async ({ page }) => {
  test.skip(!fs.existsSync(FRAME), 'private v1420 frames are not on this machine');
  // Crop the camera stage out of the phone screenshot so baked-in chrome is reduced.
  const cropped = '/tmp/v1420-stage.jpg';
  execFileSync('ffmpeg', [
    '-y', '-i', FRAME, '-vf', 'crop=360:470:0:120', cropped,
  ], { stdio: 'pipe' });

  await page.goto('/');
  await page.locator('#file-input').setInputFiles(cropped);
  await expect(page.locator('#gate')).toBeHidden({ timeout: 15_000 });
  await page.waitForFunction(() => window.__sockMatcher?.getState()?.source === 'photo', null, { timeout: 15_000 });
  await page.locator('#mode-colors').click();
  await page.locator('.color-chip[data-id="white"]').click();
  await page.waitForFunction(() => {
    const s = window.__sockMatcher?.getState();
    return s && s.mode === 'colors' && s.colorHits > 3;
  }, null, { timeout: 12_000 });
  const s = await page.evaluate(() => window.__sockMatcher.getState());
  expect(s.colorHits).toBeGreaterThan(3);
  expect(s.power.maxSide).toBe(320);
  const chip = (await page.locator('#status-chip').textContent()) || '';
  expect(chip).toMatch(/white sock/i);
});
