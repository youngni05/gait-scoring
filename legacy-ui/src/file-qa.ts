import { BrowserVideoSource } from '../../mobile/src/pose/browser-video-source';
const output = document.querySelector<HTMLPreElement>('#result')!;
const mount = document.querySelector<HTMLDivElement>('#preview')!;
const assert = (ok: unknown, message: string) => { if (!ok) throw new Error(message); };
document.querySelector<HTMLButtonElement>('#run')!.onclick = async () => {
 const button = document.querySelector<HTMLButtonElement>('#run')!; button.disabled = true;
 output.textContent = '';
 const log = (text: string) => { output.textContent += text + '\n'; };
 let session: Awaited<ReturnType<BrowserVideoSource['open']>> | undefined;
 try {
  const files = [...document.querySelector<HTMLInputElement>('#files')!.files!];
  const file = files.find(f => f.name.endsWith('.mp4'))!;
  const expected: number[] = JSON.parse(await files.find(f => f.name.endsWith('.json'))!.text());
  assert(expected.length > 0, 'Empty reference');
  session = await new BrowserVideoSource(mount).open(file, new AbortController().signal);
  for (const delay of [0, 40]) {
   let count = 0, maxError = 0;
   for await (const frame of session.frames(new AbortController().signal)) {
    try {
     assert(count < expected.length, 'Extra output frame');
     maxError = Math.max(maxError, Math.abs(frame.t - expected[count]!));
     assert(maxError < 1e-8, `PTS mismatch at ${count}`);
     const canvas = frame.pixels as HTMLCanvasElement;
     assert(canvas.width === session.info.width && canvas.height === session.info.height, 'Pixel buffer size mismatch');
     count++;
     if (delay) await new Promise(resolve => setTimeout(resolve, delay));
    } finally { frame.release(); }
   }
   assert(count === expected.length && session.verification?.complete, 'Incomplete output');
   log(`PASS delay=${delay}ms: ${count}/${expected.length}, ALL PTS maxError=${maxError}s, canvas=${session.info.width}x${session.info.height}`);
  }
  const abort = new AbortController(); let count = 0, cancelled = false;
  try { for await (const frame of session.frames(abort.signal)) { frame.release(); if (++count === 17) abort.abort(); } }
  catch (e) { cancelled = e instanceof DOMException && e.name === 'AbortError'; }
  assert(cancelled && count === 17 && !session.verification?.complete, 'Cancel did not stop at 17');
  let restarted = 0;
  for await (const frame of session.frames(new AbortController().signal)) {
   try { assert(Math.abs(frame.t - expected[restarted]!) < 1e-8, 'Restart PTS mismatch'); restarted++; } finally { frame.release(); }
  }
  assert(restarted === expected.length && session.verification?.complete, 'Restart failed');
  log(`PASS cancel at 17; restart ${restarted}/${expected.length}, ALL PTS matched`);
  log(`First=${expected[0]}s; last=${expected.at(-1)}s`);
  await session.dispose(); assert(mount.childElementCount === 0, 'Cleanup failed');
  log('PASS cleanup; ALL CHECKS PASSED');
 } catch (error) { log(`FAIL ${error instanceof Error ? error.message : error}`); }
 finally { await session?.dispose(); button.disabled = false; }
};
