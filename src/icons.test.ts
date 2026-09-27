import { test, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { icon, ICON_NAMES } from './icons';

test('every icon renders an svg', () => {
  expect(ICON_NAMES.length).toBe(16);
  for (const name of ICON_NAMES) {
    const svg = icon(name);
    expect(svg).toContain('viewBox="0 0 24 24"');
    expect(svg).toContain('stroke="currentColor"');
    expect(svg).toContain('aria-hidden="true"');
    expect(svg).toMatch(/<(path|circle|rect|polyline|line|polygon)\b/);
  }
});

test('no backdrop-filter over the canvas', () => {
  expect(readFileSync(new URL('./app.css', import.meta.url), 'utf8')).not.toContain('backdrop-filter');
});
