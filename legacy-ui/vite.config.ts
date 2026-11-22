import { defineConfig } from 'vite';

export default defineConfig({
  appType: 'mpa',
  publicDir: '../mobile/public',
  server: { port: 5180, strictPort: true },
});
