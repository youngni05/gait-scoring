import { readdir, readFile } from 'node:fs/promises';
const root = new URL('../mobile/dist/', import.meta.url);
async function check(dir) {
  for (const item of await readdir(dir, {withFileTypes:true})) {
    const path = new URL(item.name + (item.isDirectory() ? '/' : ''), dir);
    if (item.isDirectory()) await check(path);
    else if (/\.(js|html)$/.test(item.name)) {
      const content = await readFile(path, 'utf8');
      if (content.includes('합성 좌표 준비') || content.includes('합성 좌표 자동 테스트') || item.name.includes('SyntheticMeasurement')) throw new Error('E2E control leaked into production build');
    }
  }
}
await check(root);
console.log('Production output excludes synthetic measurement controls.');
