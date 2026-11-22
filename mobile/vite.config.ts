import { defineConfig } from 'vite';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

// Fingerprint the exact loader and binary shipped by the locked MediaPipe package.
const resolvePackage = createRequire(import.meta.url).resolve;
function runtimeAsset(name: string) {
  const bytes = readFileSync(resolvePackage('@mediapipe/tasks-vision/' + name));
  return {path:'/vendor/mediapipe/' + name, bytes:bytes.length, sha256:createHash('sha256').update(bytes).digest('hex')};
}

export default defineConfig({
  appType: 'mpa',
  define: {
    __POSE_RUNTIME__: JSON.stringify({
      loader: runtimeAsset('vision_wasm_internal.js'),
      wasm: runtimeAsset('vision_wasm_internal.wasm'),
    }),
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        rewrite: path => path.replace(/^\/api/, ''),
      },
    },
  },
});
