/** Integration example: local preparation only; caller explicitly chooses a gateway. */
import {BrowserVideoSource} from './browser-video-source';
import {MediaPipePoseExtractor} from './mediapipe-extractor';
import {extractVideo} from './extract-video';
import {FrontalStandDetector} from './stand-detector';
import {JsonCoordsSerializer} from './coords-serializer';
import type {UploadRequest} from './upload-gateway';

export async function prepareRecording(file: File, mount: HTMLElement,
  input: {subjectId: string; heightCm: number; analysisStartedAt: string; recordingId: string; requestId: string},
  signal: AbortSignal): Promise<UploadRequest> {
  const session = await new BrowserVideoSource(mount).open(file, signal);
  let engine: MediaPipePoseExtractor | undefined;
  try {
    // Internal coords-1 serializer adapter. measurement-2 removes these legacy fields before upload.
    // Video capture metadata is deliberately not used for the analysis date.
    const date = { recordedAt: input.analysisStartedAt, recordedAtSource: 'user_entered' as const };
    engine = await MediaPipePoseExtractor.create(signal);
    const run = await extractVideo(session, engine, signal);
    if (!session.verification?.complete) throw Error('전체 프레임 검증 미완료');
    const detection = new FrontalStandDetector(session.info.width/session.info.height)
      .analyze(run.frames, session.info.durationSeconds, signal).detection;
    const coords = new JsonCoordsSerializer().create({
      input: {...date, subjectId: input.subjectId, heightCm: input.heightCm, recordingId: input.recordingId},
      video: session.info, frames: run.frames, marks: detection.marks, pose: engine.configuration,
    });
    if (!coords.ok) throw Error(coords.issues.map(i=>i.message).join('; '));
    return {coords: coords.value, requestId: input.requestId, metadata: {
      recordedAtSource: date.recordedAtSource, rotationAppliedDegrees: session.info.rotationAppliedDegrees,
      detection, modelAssetVersion: 'pose_landmarker_full/float16/1',
    }};
  } finally { try { await engine?.dispose(); } finally { await session.dispose(); } }
}
