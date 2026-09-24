import { appendFileSync, mkdirSync } from 'node:fs';
import type { Page } from '@playwright/test';

/** Follow a GPU test run live with: tail -f test-results/progress.log */
export const PROGRESS_LOG = 'test-results/progress.log';

export function streamProgress(page: Page, label: string) {
  mkdirSync('test-results', { recursive: true });
  const write = (line: string) => {
    const out = `${new Date().toLocaleTimeString()} ${line}`;
    console.log(out);
    appendFileSync(PROGRESS_LOG, out + '\n');
  };
  write(`=== ${label}`);
  page.on('console', (m) => write(m.text()));
  page.on('pageerror', (e) => write(`PAGE ERROR ${e.message}`));
}
