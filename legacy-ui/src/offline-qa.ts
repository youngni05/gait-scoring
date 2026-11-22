import { BrowserVideoSource } from '../../mobile/src/pose/browser-video-source';
const output = document.querySelector<HTMLPreElement>('#result')!;
const mount = document.querySelector<HTMLDivElement>('#preview')!;
const assert = (condition: unknown, message: string) => { if (!condition) throw new Error(message); };
document.querySelector<HTMLButtonElement>('#run')!.onclick = async () => {
  output.textContent = '';
  const files = [...document.querySelector<HTMLInputElement>('#files')!.files!];
  const log = (s: string) => { output.textContent += s + '\n'; };
  try {
    const videos = files.filter(f => f.name.endsWith('.mp4') && f.name !== 'broken.mp4');
    assert(videos.length >= 3, 'Select at least three fixtures');
    for (const file of videos) {
      const expectedFile = files.find(f => f.name === file.name.replace('.mp4', '.expected.json'));
      assert(expectedFile, 'Missing independent PyAV expected timestamps');
      const expected: number[] = JSON.parse(await expectedFile!.text());
      const session = await new BrowserVideoSource(mount).open(file, new AbortController().signal);
      try {
        for (const delay of [0, 40]) {
          let count = 0, maxError = 0;
          for await (const frame of session.frames(new AbortController().signal)) {
            try {
              assert(count < expected.length, 'Extra frame');
              maxError = Math.max(maxError, Math.abs(frame.t - expected[count]!));
              assert(maxError < 1e-8, `PTS mismatch at ${count}`);
              // Fixture contains a red rectangle moving three pixels per source frame.
              const canvas = frame.pixels as HTMLCanvasElement;
              const row = canvas.getContext('2d')!.getImageData(0, 40, canvas.width, 1).data;
              let sum = 0, n = 0;
              for (let x = 0; x < canvas.width; x++) if (row[x*4]! > 150) { sum += x; n++; }
              const center = (count * 3) % (canvas.width - 60) + 29.5;
              assert(n > 0 && Math.abs(sum/n - center) <= 1, `Pixel content/order mismatch at ${count}`);
              count++;
              if (delay) await new Promise(r => setTimeout(r, delay));
            } finally { frame.release(); }
          }
          assert(count === expected.length && session.verification?.complete, 'Missing frames or incomplete verification');
          log(`PASS ${file.name} delay=${delay}ms: ${count}/${expected.length}, ALL PTS maxError=${maxError}s, ALL pixel markers matched`);
        }
        const abort = new AbortController(); let count = 0, cancelled = false;
        try { for await (const frame of session.frames(abort.signal)) { frame.release(); if (++count === 5) abort.abort(); } }
        catch (error) { cancelled = error instanceof DOMException && error.name === 'AbortError'; }
        assert(cancelled && !session.verification?.complete && count === 5, 'Cancellation incorrectly completed');
        let restarted = 0;
        for await (const frame of session.frames(new AbortController().signal)) { frame.release(); restarted++; }
        assert(restarted === expected.length && session.verification?.complete, 'Restart did not recover');
        log(`PASS ${file.name}: cancel at 5 and restart ${restarted}/${expected.length}`);
      } finally { await session.dispose(); }
      assert(mount.childElementCount === 0, 'Preview cleanup failed');
    }
    const probe = videos[0]!;
    const early = new AbortController();
    const opening = new BrowserVideoSource(mount).open(probe, early.signal); early.abort();
    let earlyCancelled = false;
    try { await opening; } catch (e) { earlyCancelled = e instanceof DOMException && e.name === 'AbortError'; }
    assert(earlyCancelled && mount.childElementCount === 0, 'Immediate open cancellation leaked resources');
    const disposed = await new BrowserVideoSource(mount).open(probe, new AbortController().signal);
    const iterator = disposed.frames(new AbortController().signal)[Symbol.asyncIterator]();
    const pending = iterator.next(); await disposed.dispose();
    let disposeCancelled = false;
    try { await pending; } catch (e) { disposeCancelled = e instanceof DOMException && e.name === 'AbortError'; }
    assert(disposeCancelled && !disposed.verification?.complete, 'Dispose did not cancel pending read');
    await disposed.dispose();
    const concurrent = await new BrowserVideoSource(mount).open(probe, new AbortController().signal);
    const active = concurrent.frames(new AbortController().signal)[Symbol.asyncIterator]();
    const first = await active.next(); first.value?.release();
    let refused = false;
    try { await concurrent.frames(new AbortController().signal)[Symbol.asyncIterator]().next(); } catch { refused = true; }
    assert(refused, 'Concurrent iteration accepted'); await active.return?.();
    assert(!concurrent.verification?.complete, 'Early iterator return marked complete');
    let recovered = 0; for await (const frame of concurrent.frames(new AbortController().signal)) { frame.release(); recovered++; }
    assert(recovered > 0 && concurrent.verification?.complete, 'Early return did not recover');
    await concurrent.dispose();
    log('PASS immediate open cancel; dispose during pending read; concurrent rejection; early return and recovery');
    const broken = files.find(f => f.name === 'broken.mp4');
    assert(broken, 'Select broken.mp4'); let rejected = false;
    try { await new BrowserVideoSource(mount).open(broken!, new AbortController().signal); } catch { rejected = true; }
    assert(rejected && mount.childElementCount === 0, 'Broken input accepted or leaked preview');
    log('PASS broken input rejected; preview removed after every run');
    log('ALL CHECKS PASSED');
  } catch (error) { log(`FAIL: ${error instanceof Error ? error.stack : String(error)}`); }
};
