import type { PoseFrame } from '../../../contracts/coords-1/types';
import type { PoseExtractor, VideoSession } from './ports';
export interface PoseRun { frames: PoseFrame[]; detected: number; missing: number; elapsedMs: number }
export async function extractVideo(session: VideoSession, extractor: PoseExtractor, signal: AbortSignal,
  progress: (count: number, detected: number) => void = () => {}): Promise<PoseRun> {
  const frames: PoseFrame[] = []; let detected = 0; const start = performance.now();
  for await (const frame of session.frames(signal)) {
    try {
      if (signal.aborted) throw new DOMException('취소됨', 'AbortError');
      const pose = await extractor.extract(frame, signal);
      if (pose.t !== frame.t || (frames.length && pose.t <= frames.at(-1)!.t)) throw new Error('관절 출력 시각 불일치');
      frames.push(pose); if (pose.lm) detected++;
      progress(frames.length, detected);
    } finally { frame.release(); }
    // Yield to input events between synchronous VIDEO inferences; never skip a frame.
    await new Promise(resolve => setTimeout(resolve, 0));
    if (signal.aborted) throw new DOMException('취소됨', 'AbortError');
  }
  if (signal.aborted) throw new DOMException('취소됨', 'AbortError');
  return { frames, detected, missing: frames.length - detected, elapsedMs: performance.now() - start };
}
