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

test('none shows the Re note when tau clamps', async ({ page }) => {
  streamProgress(page, 'tunnel3d none clamp');
  await page.goto('/3d.html');
  await page.waitForFunction(() => (window as unknown as { tunnel3d?: { ready: boolean } }).tunnel3d?.ready === true, null, { timeout: 60_000 });
  await page.click('.menu-button');
  await page.selectOption('#obstacle', 'none');
  await page.fill('#re', '1000000');
  await expect(page.locator('#reNote')).toContainText('needs τ below 0.51');
});

// With a fixed TRT Lambda of 3/16 the inlet drove odd modes unstable below tau 0.535 (Re 200 here).
test('sphere stays stable at Re 400', async ({ page }) => {
  streamProgress(page, 'tunnel3d Re 400');
  await page.goto('/3d.html');
  await page.waitForFunction(() => (window as unknown as { tunnel3d?: { ready: boolean } }).tunnel3d?.ready === true, null, { timeout: 60_000 });
  await page.click('.menu-button');
  // FP32, because an FP16 blow-up clamps to finite noise instead of NaN.
  await page.evaluate(() => {
    const s = document.getElementById('precision') as HTMLSelectElement;
    s.value = 'fp32';
    s.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.waitForFunction(() => (window as unknown as { tunnel3d: { step(): number } }).tunnel3d.step() < 1000, null, { timeout: 60_000 });
  await page.fill('#re', '400');
  await page.waitForFunction(() => document.querySelector('.toast:not([hidden])') !== null || (window as unknown as { tunnel3d: { step(): number } }).tunnel3d.step() > 12_000, null, { timeout: 120_000 });
  await expect(page.locator('.toast:not([hidden])')).toHaveCount(0);
  expect(Number.isFinite(await page.evaluate(() => (window as unknown as { tunnel3d: { cd(): number } }).tunnel3d.cd()))).toBe(true);
});

test('tracers follow the colour-by field, even while paused', async ({ page }) => {
  streamProgress(page, 'tunnel3d tracer colour');
  await page.goto('/3d.html');
  await page.waitForFunction(() => (window as unknown as { tunnel3d?: { ready: boolean } }).tunnel3d?.ready === true, null, { timeout: 60_000 });
  await page.waitForFunction(() => (window as unknown as { tunnel3d: { step(): number } }).tunnel3d.step() > 4000, null, { timeout: 60_000 });
  const set = (id: string, value: string) =>
    page.evaluate(([id, value]) => {
      const e = document.getElementById(id) as HTMLSelectElement;
      e.value = value;
      e.dispatchEvent(new Event('change', { bubbles: true }));
    }, [id, value]);
  await set('sliceAxis', 'off');
  await page.click('.menu-button');
  await page.click('#pause');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);
  const speed = await page.locator('#view').screenshot();
  await set('viewMode', '1');
  await page.waitForTimeout(500);
  const vorticity = await page.locator('#view').screenshot();
  expect(vorticity.equals(speed)).toBe(false);
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
