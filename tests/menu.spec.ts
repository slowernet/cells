import { test, expect, type Page } from '@playwright/test';
import { streamProgress } from './progress';

const pages = ['/index.html', '/3d.html'];
const viewports = [
  { name: 'desktop', viewport: { width: 1400, height: 800 }, hasTouch: false, isMobile: false },
  { name: 'phone', viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true },
];

const canvasSize = (page: Page) => page.evaluate(() => {
  const c = document.getElementById('view') as HTMLCanvasElement;
  return [c.width, c.height];
});

const step = (page: Page) => page.evaluate(() => Number((document.getElementById('step')!.textContent ?? '0').replace(/\D/g, '')));

for (const url of pages)
  for (const vp of viewports)
    test.describe(`${url} ${vp.name}`, () => {
      test.use({ viewport: vp.viewport, hasTouch: vp.hasTouch, isMobile: vp.isMobile });

      test('menu opens, closes and scrolls', async ({ page }) => {
        streamProgress(page, `menu ${url} ${vp.name}`);
        await page.goto(url);
        await page.waitForFunction(() => Number((document.getElementById('step')?.textContent ?? '0').replace(/\D/g, '')) > 0, null, { timeout: 60_000 });
        const before = await canvasSize(page);
        expect(before[0]).toBeGreaterThan(0);

        const panel = page.locator('#panel');
        await expect(panel).toBeHidden();
        await page.click('.menu-button');
        await expect(panel).toBeVisible();
        await expect(page.locator('.menu-button')).toHaveAttribute('aria-expanded', 'true');
        // Wait for the 180 ms slide-in to finish, so gestures land on the panel and not the canvas behind it.
        await expect.poll(() => panel.evaluate((el) => el.getBoundingClientRect().left)).toBe(0);

        // Open every group so the content is taller than the viewport, then scroll.
        await page.evaluate(() => document.querySelectorAll<HTMLDetailsElement>('#panel details').forEach((d) => (d.open = true)));
        const dims = await panel.evaluate((el) => [el.scrollHeight, el.clientHeight]);
        expect(dims[0]).toBeGreaterThan(dims[1]);
        if (vp.hasTouch) {
          // A touch drag inside the panel, in CSS pixels of the 390x844 viewport.
          const cdp = await page.context().newCDPSession(page);
          await cdp.send('Input.synthesizeScrollGesture', { x: 180, y: 600, yDistance: -300, gestureSourceType: 'touch' });
        } else {
          await panel.hover();
          await page.mouse.wheel(0, 400);
        }
        await expect.poll(() => panel.evaluate((el) => el.scrollTop)).toBeGreaterThan(0);

        await page.keyboard.press('Escape');
        await expect(panel).toBeHidden();
        await expect(page.locator('.menu-button')).toHaveAttribute('aria-expanded', 'false');
        expect(await canvasSize(page)).toEqual(before);
      });

      test('controls still work', async ({ page }) => {
        await page.goto(url);
        await page.waitForFunction(() => Number((document.getElementById('step')?.textContent ?? '0').replace(/\D/g, '')) > 0, null, { timeout: 60_000 });
        await page.click('.menu-button');
        await page.evaluate(() => document.querySelectorAll<HTMLDetailsElement>('#panel details').forEach((d) => (d.open = true)));
        await page.selectOption('#obstacle', url === '/3d.html' ? 'cube' : 'square');
        const s0 = await step(page);
        await expect.poll(() => step(page), { timeout: 10_000 }).toBeGreaterThan(s0);
      });
    });
