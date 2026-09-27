import { test, expect } from '@playwright/test';
import { streamProgress } from './progress';

test('smoke', async ({ page }) => {
  streamProgress(page, 'tunnel3d smoke');
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/3d.html');
  await page.waitForFunction(() => (window as unknown as { tunnel3d?: { ready: boolean } }).tunnel3d?.ready === true, null, { timeout: 60_000 });
  await page.waitForTimeout(5000);
  const cd = await page.evaluate(() => (window as unknown as { tunnel3d: { cd(): number } }).tunnel3d.cd());
  const pixels = await page.evaluate(() => (window as unknown as { tunnel3d: { pixelCount(): Promise<number> } }).tunnel3d.pixelCount());
  console.log(`tunnel3d smoke: C_D ${cd}, ${pixels} pixels drawn`);
  expect(Number.isFinite(cd) && cd > 0, `C_D ${cd}`).toBe(true);
  expect(pixels).toBeGreaterThan(1000);
  expect(errors).toEqual([]);
});

test('throughput', async ({ page }) => {
  test.skip(!process.env.THROUGHPUT, 'set THROUGHPUT=1 on the reference machine with the GPU idle');
  streamProgress(page, 'tunnel3d throughput');
  await page.goto('/3d.html');
  await page.waitForFunction(() => (window as unknown as { tunnel3d?: { ready: boolean } }).tunnel3d?.ready === true, null, { timeout: 60_000 });
  await page.waitForFunction(() => (window as unknown as { tunnel3d: { step(): number } }).tunnel3d.step() >= 3000, null, { timeout: 60_000 });
  const step = () => page.evaluate(() => (window as unknown as { tunnel3d: { step(): number } }).tunnel3d.step());
  const s0 = await step();
  const t0 = Date.now();
  await page.waitForTimeout(10_000);
  const rate = ((await step()) - s0) / ((Date.now() - t0) / 1000);
  console.log(`tunnel3d steps/s ${rate.toFixed(0)}`);
  expect(rate).toBeGreaterThanOrEqual(480);
});
