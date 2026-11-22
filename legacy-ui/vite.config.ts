import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  appType: 'mpa',
  build: { rolldownOptions: { input: { index: fileURLToPath(new URL('./index.html', import.meta.url)), design: fileURLToPath(new URL('./design.html', import.meta.url)) } } },
  publicDir: '../mobile/public',
  server: { port: 5180, strictPort: true },
});
