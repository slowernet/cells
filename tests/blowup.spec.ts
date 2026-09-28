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

  const resets = () => page.evaluate(() => (window as unknown as { flowResets: () => number }).flowResets());
  const toast = page.locator('.toast');
  await expect(toast).toBeVisible({ timeout: 30_000 });
  await expect(toast).toContainText('The flow became unstable');
  // The first blow-up resets the flow and keeps the drawing.
  expect(await resets()).toBeGreaterThanOrEqual(1);
  expect(await page.evaluate(() => (window as unknown as { interactionState: () => string }).interactionState())).toBe(drawn);

  // The same setup blows up again; the page resets once more and pauses instead of looping.
  await expect(page.locator('#pause')).toHaveAttribute('aria-label', 'Run', { timeout: 30_000 });
  await expect(toast).toContainText("so it's paused");
  expect(await resets()).toBeGreaterThanOrEqual(2);
  // The reset happens just before the pause, so the paused solver sits at step 0; without a reset it keeps its pre-blow-up step.
  const solverStep = () => page.evaluate(() => (window as unknown as { solverStep: () => number }).solverStep());
  expect(await solverStep()).toBe(0);
  await page.waitForTimeout(1000);
  expect(await solverStep()).toBe(0);
  await expect(toast).toBeVisible();
});

test('2D with no obstacle stays stable at the default Re', async ({ page }) => {
  streamProgress(page, 'blowup 2D none stable');
  await page.goto('/index.html');
  await page.waitForFunction(() => Number((document.getElementById('step')?.textContent ?? '0').replace(/\D/g, '')) > 0, null, { timeout: 60_000 });
  await page.evaluate(() => {
    const s = document.getElementById('obstacle') as HTMLSelectElement;
    s.value = 'none';
    s.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await expect(page.locator('#tauNote')).not.toContainText('L = 1 cells');
  await page.waitForTimeout(10_000);
  expect(await page.evaluate(() => (window as unknown as { flowResets: () => number }).flowResets())).toBe(0);
});

test('2D with no obstacle still detects a blow-up through the field check', async ({ page }) => {
  streamProgress(page, 'blowup 2D none field check');
  await page.goto('/index.html');
  await page.waitForFunction(() => Number((document.getElementById('step')?.textContent ?? '0').replace(/\D/g, '')) > 0, null, { timeout: 60_000 });
  await page.evaluate(() => {
    const s = document.getElementById('obstacle') as HTMLSelectElement;
    s.value = 'none';
    s.dispatchEvent(new Event('change', { bubbles: true }));
    // Re far past the ceiling clamps tau at 0.51, which is unstable with nothing to measure a force on.
    const re = document.getElementById('re') as HTMLInputElement;
    re.value = '1000000';
    re.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect(page.locator('.toast')).toBeVisible({ timeout: 30_000 });
  expect(await page.evaluate(() => (window as unknown as { flowResets: () => number }).flowResets())).toBeGreaterThanOrEqual(1);
});
