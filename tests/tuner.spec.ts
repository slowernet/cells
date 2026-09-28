import { test, expect } from '@playwright/test';

// Auto steps-per-frame must climb back after dropping to 1; round(1 * 1.1) = 1 used to pin it there.
test('2D auto steps per frame recovers from 1', async ({ page }) => {
  await page.goto('/index.html');
  await page.waitForFunction(() => Number(document.getElementById('spfOut')?.textContent) > 0, null, { timeout: 60_000 });
  const setMode = (v: string) =>
    page.evaluate((value) => {
      const s = document.getElementById('spfMode') as HTMLSelectElement;
      s.value = value;
      s.dispatchEvent(new Event('change'));
    }, v);
  await setMode('1');
  await page.waitForTimeout(500);
  await setMode('auto');
  await expect.poll(() => page.evaluate(() => Number(document.getElementById('spfOut')!.textContent)), { timeout: 5000 }).toBeGreaterThan(4);
});
