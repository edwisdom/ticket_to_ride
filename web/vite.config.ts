import { svelte } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// `make web-dev` serves the client here and proxies the API to `ttr serve`.
const server = process.env.TTR_SERVER ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [svelte()],
  build: {
    // FastAPI serves the build; Node is a build-time tool only.
    outDir: '../ticket_to_ride/ui/web/static',
    emptyOutDir: true,
  },
  server: {
    // The development board gallery imports the generated layouts directly.
    fs: { allow: ['.', '../ticket_to_ride/data/layout'].map((p) => fileURLToPath(new URL(p, import.meta.url))) },
    proxy: {
      '/api': server,
      '/ws': { target: server.replace(/^http/, 'ws'), ws: true },
    },
  },
  test: { include: ['src/**/*.test.ts'] },
});
