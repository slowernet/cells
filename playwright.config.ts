import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'tests',
  timeout: 15 * 60_000,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:5179',
    channel: 'chrome',
    launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] },
  },
  webServer: {
    // A static build, so editing sources mid-run cannot hot-reload the page under test.
    command: 'npx vite build --outDir dist-test && npx vite preview --outDir dist-test --port 5179 --strictPort',
    url: 'http://localhost:5179',
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
