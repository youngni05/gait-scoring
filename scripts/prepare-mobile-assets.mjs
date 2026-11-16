import { mkdir, cp, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const root = new URL('../', import.meta.url);
const model = new URL('mobile/public/models/pose_landmarker_full.task', root);
const manifest = JSON.parse(await readFile(new URL('mobile/model-assets.json', root), 'utf8'));
await mkdir(new URL('mobile/public/models/', root), { recursive: true });
let bytes;
try { bytes = await readFile(model); } catch {
  const response = await fetch(manifest.url);
  if (!response.ok) throw new Error(`Model download failed: ${response.status}`);
  bytes = Buffer.from(await response.arrayBuffer());
}
const sha = createHash('sha256').update(bytes).digest('hex');
if (sha !== manifest.sha256) throw new Error(`Model checksum mismatch: ${sha}`);
await writeFile(model, bytes);
await cp(new URL('node_modules/@mediapipe/tasks-vision/wasm/', root),
  new URL('mobile/public/vendor/mediapipe/', root), { recursive: true });
console.log('Pinned model and npm-locked WASM assets ready locally. No user media read.');
