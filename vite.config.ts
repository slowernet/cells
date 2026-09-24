/// <reference types="vitest/config" />
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    target: 'es2022',
    rolldownOptions: {
      input: { main: 'index.html', bench: 'bench.html', validate: 'validate.html' },
    },
  },
  test: { include: ['src/**/*.test.ts'] },
});
