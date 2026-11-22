import model from '../../../../ai-module/pose-asset.json';

interface Asset { path:string; bytes:number; sha256:string }
declare const __POSE_RUNTIME__: {loader:Asset; wasm:Asset};
const cacheName = 'happywalk-public-pose-v1';

async function matches(bytes:ArrayBuffer, asset:Asset) {
  if (bytes.byteLength !== asset.bytes) return false;
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(hash, b => b.toString(16).padStart(2,'0')).join('') === asset.sha256;
}

async function loadAsset(asset:Asset, cache:Cache | undefined, signal:AbortSignal, received:(bytes:number)=>void) {
  const url = new URL(asset.path, location.origin);
  const hit = await cache?.match(url);
  if (hit?.headers.get('X-Asset-SHA256') === asset.sha256) {
    const bytes = await hit.arrayBuffer();
    if (await matches(bytes, asset)) return bytes;
  }
  signal.throwIfAborted();
  url.searchParams.set('v',asset.sha256);
  const response = await fetch(url,{signal,cache:'no-store',credentials:'omit',redirect:'error'});
  if (!response.ok || !response.body) throw Error('분석 파일을 받을 수 없습니다. 연결을 확인하고 다시 시도해 주세요.');
  const reader = response.body.getReader();
  const bytes = new Uint8Array(asset.bytes);
  let count = 0;
  received(count);
  try {
    for (;;) {
      const {value,done} = await reader.read();
      if (done) break;
      if (count + value.length > bytes.length) throw Error('분석 파일의 크기가 예상과 다릅니다.');
      bytes.set(value,count);
      count += value.length;
      received(count);
    }
  } finally {
    await reader.cancel().catch(()=>{});
    reader.releaseLock();
  }
  if (count !== asset.bytes || !await matches(bytes.buffer,asset))
    throw Error('분석 파일이 완전히 다운로드되지 않았거나 버전이 다릅니다. 다시 시도해 주세요.');
  signal.throwIfAborted();
  if (cache) {
    // One canonical entry per public file; only a verified complete version replaces it.
    await cache.put(new URL(asset.path, location.origin),new Response(bytes,{headers:{'X-Asset-SHA256':asset.sha256}})).catch(()=>{});
  }
  return bytes.buffer;
}

export async function preparePoseAssets(apiBase:string, signal:AbortSignal, progress:(detail:string, percent:number)=>void) {
  let cache: Cache | undefined;
  try { cache = await caches.open(cacheName); } catch { /* Private browsing can restrict persistent storage. */ }
  const task = {path:apiBase + '/assets/pose-full.task', ...model};
  async function get(asset:Asset,label:string) {
    let last = -Infinity;
    return loadAsset(asset,cache,signal,count=>{
      const now = performance.now();
      if (count < asset.bytes && now - last < 100) return;
      last = now;
      progress(label+' 받는 중 · '+(count/1000000).toFixed(1)+' / '+(asset.bytes/1000000).toFixed(1)+' MB', Math.floor(100*count/asset.bytes));
    });
  }
  const modelBytes = await get(task,'관절 모델');
  const loader = await get(__POSE_RUNTIME__.loader,'실행 스크립트');
  const wasm = await get(__POSE_RUNTIME__.wasm,'실행 파일');
  signal.throwIfAborted();
  const files = {
    wasmLoaderPath:URL.createObjectURL(new Blob([loader],{type:'text/javascript'})),
    wasmBinaryPath:URL.createObjectURL(new Blob([wasm],{type:'application/wasm'})),
  };
  return {modelBytes:new Uint8Array(modelBytes),modelHash:model.sha256,files,
    dispose() { URL.revokeObjectURL(files.wasmLoaderPath); URL.revokeObjectURL(files.wasmBinaryPath); }};
}
