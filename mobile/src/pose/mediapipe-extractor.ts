import type { PoseLandmarker } from '@mediapipe/tasks-vision' with { 'resolution-mode': 'import' };
import type { DecodedFrame, PoseExtractor } from './ports';
import { toPoseFrame } from './pose-values';
const abortError = () => new DOMException('작업을 취소했습니다.', 'AbortError');
export class PoseInferenceError extends Error {}
export class MediaPipePoseExtractor implements PoseExtractor {
  get configuration() { return { model: 'pose_landmarker_full', engine_version: '1.2.3', delegate: this.delegate } as const; }
  private previous = -1;
  private disposed = false;
  private constructor(private engine: PoseLandmarker, readonly maxPoses: 1 | 2, readonly inputObject: 'HTMLCanvasElement' | 'ImageData', readonly delegate: 'CPU' | 'GPU') {}
  static async create(signal: AbortSignal, maxPoses: 1 | 2 = 1, inputObject: 'HTMLCanvasElement' | 'ImageData' = 'HTMLCanvasElement', modelBytes?: Uint8Array, delegate: 'CPU' | 'GPU' = 'CPU', preparedFiles?: {wasmLoaderPath:string; wasmBinaryPath:string}) {
    if (signal.aborted) throw abortError();
    const { FilesetResolver, PoseLandmarker } = await import('@mediapipe/tasks-vision');
    const files = preparedFiles ?? await FilesetResolver.forVisionTasks('/vendor/mediapipe');
    if (signal.aborted) throw abortError();
    const engine = await PoseLandmarker.createFromOptions(files, {
      baseOptions: { ...(modelBytes ? { modelAssetBuffer: modelBytes } : { modelAssetPath: '/models/pose_landmarker_full.task' }), delegate },
      runningMode: 'VIDEO', numPoses: maxPoses, outputSegmentationMasks: false,
      minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    if (signal.aborted) { engine.close(); throw abortError(); }
    return new MediaPipePoseExtractor(engine, maxPoses, inputObject, delegate);
  }
  private detect(frame: DecodedFrame, signal: AbortSignal) {
    if (signal.aborted || this.disposed) throw abortError();
    if (!Number.isFinite(frame.t) || frame.t < 0 || frame.t <= this.previous) throw new Error('MediaPipe 입력 시각은 엄격하게 증가해야 합니다.');
    if (!(frame.pixels instanceof HTMLCanvasElement)) throw new Error('Canvas 프레임이 필요합니다.');
    // API accepts milliseconds; output retains the original seconds without rounding.
    // The local team bridge records the actual input type. Legacy QA keeps its Canvas path.
    const pixels = this.inputObject === 'ImageData'
      ? frame.pixels.getContext('2d')!.getImageData(0, 0, frame.pixels.width, frame.pixels.height)
      : frame.pixels;
    let result;
    try { result = this.engine.detectForVideo(pixels, frame.t * 1000); }
    catch (cause) { throw new PoseInferenceError('관절 추출에 실패했습니다.', { cause }); }
    this.previous = frame.t;
    return result;
  }
  async extract(frame: DecodedFrame, signal: AbortSignal) {
    if (this.maxPoses !== 1) throw new Error('다인 후보 점검기는 단일 좌표 추출에 사용할 수 없습니다.');
    return toPoseFrame(frame.t, this.detect(frame, signal).landmarks);
  }
  countCandidates(frame: DecodedFrame, signal: AbortSignal): number {
    if (this.maxPoses !== 2) throw new Error('별도의 2인 후보 점검기를 사용하세요.');
    return this.detect(frame, signal).landmarks.length;
  }
  async dispose() { if (!this.disposed) { this.disposed = true; this.engine.close(); } }
}

