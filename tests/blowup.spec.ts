import { test, expect } from '@playwright/test';
import { streamProgress } from './progress';

// A wide arc upstream of the cylinder drives the flow past the lattice's limit; the page must reset and explain.
test('2D recovers from a blown-up flow and keeps the drawing', async ({ page }) => {
  streamProgress(page, 'blowup 2D');
  await page.setViewportSize({ width: 1213, height: 991 });
  await page.goto('/index.html');
  await page.waitForFunction(() => Number((document.getElementById('step')?.textContent ?? '0').replace(/\D/g, '')) > 3000, null, { timeout: 60_000 });
  const box = (await page.locator('#view').boundingBox())!;
  const pts: [number, number][] = [];
  for (let k = 0; k <= 40; k++) {
    const fy = 0.15 + (k / 40) * 0.62;
    const fx = 0.088 + 0.45 * Math.abs(fy - 0.5) ** 1.2;
    pts.push([box.x + fx * box.width, box.y + fy * box.height]);
  }
  await page.mouse.move(...pts[0]);
  await page.mouse.down();
  for (const p of pts.slice(1)) await page.mouse.move(...p, { steps: 2 });
  await page.mouse.up();
  const drawn = await page.evaluate(() => (window as unknown as { interactionState: () => string }).interactionState());
  expect(drawn).not.toBe('null');

  const toast = page.locator('.toast');
  await expect(toast).toBeVisible({ timeout: 30_000 });
  await expect(toast).toContainText('The flow became unstable');
  // The flow restarted from the inflow state with the drawing kept.
  await expect.poll(() => page.evaluate(() => Number((document.getElementById('step')!.textContent ?? '0').replace(/\D/g, '')))).toBeLessThan(20_000);
  expect(await page.evaluate(() => (window as unknown as { interactionState: () => string }).interactionState())).toBe(drawn);
  await expect.poll(() => page.evaluate(() => Number(document.getElementById('cd')!.textContent)), { timeout: 10_000 }).not.toBeNaN();
});
