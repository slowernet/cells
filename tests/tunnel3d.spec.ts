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

test('pausing keeps steps per frame and none shows no Re warning', async ({ page }) => {
  streamProgress(page, 'tunnel3d pause');
  await page.goto('/3d.html');
  await page.waitForFunction(() => (window as unknown as { tunnel3d?: { ready: boolean } }).tunnel3d?.ready === true, null, { timeout: 60_000 });
  await page.waitForTimeout(3000);
  const spf = () => page.evaluate(() => (window as unknown as { tunnel3d: { stepsPerFrame(): number } }).tunnel3d.stepsPerFrame());
  await page.click('.menu-button');
  await page.click('#pause');
  const before = await spf();
  await page.waitForTimeout(3000);
  expect(await spf()).toBe(before);
  await page.click('#pause');
  await page.selectOption('#obstacle', 'none');
  await expect(page.locator('#reNote')).toBeHidden();
});

test('slice axis off hides the slice', async ({ page }) => {
  await page.goto('/3d.html');
  await page.waitForFunction(() => (window as unknown as { tunnel3d?: { ready: boolean } }).tunnel3d?.ready === true, null, { timeout: 60_000 });
  await page.waitForTimeout(1000);
  const pixels = () => page.evaluate(() => (window as unknown as { tunnel3d: { pixelCount(): Promise<number> } }).tunnel3d.pixelCount());
  await page.click('.menu-button');
  await page.uncheck('#tracers');
  await page.keyboard.press('Escape');
  const withSlice = await pixels();
  await page.click('.menu-button');
  await page.selectOption('#sliceAxis', 'off');
  await page.keyboard.press('Escape');
  const without = await pixels();
  console.log(`pixels with slice ${withSlice}, without ${without}`);
  // The slice is the largest drawn surface; without it only the outline and the obstacle remain.
  expect(without).toBeLessThan(withSlice * 0.5);
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
