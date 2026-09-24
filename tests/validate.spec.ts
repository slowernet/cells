import { test, expect } from '@playwright/test';
import { streamProgress } from './progress';

const only = process.env.CASES?.split(',');
const all = ['poiseuille', 'periodicSeam', 'taylorGreen', 'schaferTurek1', 'schaferTurek2', 'unconfinedCylinder', 'cavity', 'naca'];

for (const key of only ?? all) {
  test(key, async ({ page }) => {
    streamProgress(page, `validate ${key}`);
    await page.goto('/validate.html');
    await page.waitForFunction(() => (window as any).validateReady === true);
    const r = await page.evaluate((k) => (window as any).runCase(k), key);
    expect(r.pass, JSON.stringify(r.metrics)).toBe(true);
  });
}
