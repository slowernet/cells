import { test } from '@playwright/test';
import { streamProgress } from './progress';

test('mlups benchmark', async ({ page }) => {
  streamProgress(page, 'benchmark');
  await page.goto('/bench.html');
  await page.waitForFunction(() => (window as any).benchReady === true);
  await page.evaluate(() => (window as any).runBenchmark());
});
