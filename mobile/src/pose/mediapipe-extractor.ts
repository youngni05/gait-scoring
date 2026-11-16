import type { PoseLandmarker } from '@mediapipe/tasks-vision' with { 'resolution-mode': 'import' };
import type { DecodedFrame, PoseExtractor } from './ports';
import { toPoseFrame } from './pose-values';
const abortError = () => new DOMException('작업을 취소했습니다.', 'AbortError');
export class MediaPipePoseExtractor implements PoseExtractor {
  readonly configuration = { model: 'pose_landmarker_full', engine_version: '1.2.3', delegate: 'CPU' } as const;
  private previous = -1;
  private disposed = false;
  private constructor(private engine: PoseLandmarker, readonly maxPoses: 1 | 2) {}
  static async create(signal: AbortSignal, maxPoses: 1 | 2 = 1) {
    if (signal.aborted) throw abortError();
    const { FilesetResolver, PoseLandmarker } = await import('@mediapipe/tasks-vision');
    const files = await FilesetResolver.forVisionTasks('/vendor/mediapipe');
    if (signal.aborted) throw abortError();
    const engine = await PoseLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: '/models/pose_landmarker_full.task', delegate: 'CPU' },
      runningMode: 'VIDEO', numPoses: maxPoses, outputSegmentationMasks: false,
      minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    if (signal.aborted) { engine.close(); throw abortError(); }
    return new MediaPipePoseExtractor(engine, maxPoses);
  }
  private detect(frame: DecodedFrame, signal: AbortSignal) {
    if (signal.aborted || this.disposed) throw abortError();
    if (!Number.isFinite(frame.t) || frame.t < 0 || frame.t <= this.previous) throw new Error('MediaPipe 입력 시각은 엄격하게 증가해야 합니다.');
    if (!(frame.pixels instanceof HTMLCanvasElement)) throw new Error('Canvas 프레임이 필요합니다.');
    // API accepts milliseconds; output retains the original seconds without rounding.
    const result = this.engine.detectForVideo(frame.pixels, frame.t * 1000);
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

