// Compile-only contract assertions. No real video, engine or upload implementation.
import type { Coords1, StandMarks } from '../contracts/coords-1/types';
import type { ProcessingInput, StandDetection, UploadGateway, VideoSource,
  PoseExtractor, StandDetector, CoordsSerializer } from '../mobile/src/pose/ports';

export async function checkPorts(source: VideoSource, extractor: PoseExtractor,
  detector: StandDetector, serializer: CoordsSerializer, gateway: UploadGateway,
  input: ProcessingInput, signal: AbortSignal): Promise<void> {
  const session = await source.open(undefined, signal);
  try {
    for await (const decoded of session.frames(signal)) {
      try { await extractor.extract(decoded, signal); }
      finally { decoded.release(); }
    }
    const detection = await detector.detect([], session.info.durationSeconds, signal);
    const result = serializer.create({ input, video: session.info,
      pose: extractor.configuration, frames: [], marks: detection.marks });
    if (result.ok) {
      serializer.stringify(result.value);
      await gateway.submit({ coords: result.value, requestId: 'compile-only',
        metadata: { recordedAtSource: input.recordedAtSource,
          rotationAppliedDegrees: session.info.rotationAppliedDegrees, detection } }, signal);
    }
  } finally { await session.dispose(); await extractor.dispose(); }
}

export function checkRejectedTypes(coords: Coords1): void {
  // @ts-expect-error Original coordinate arrays are readonly.
  coords.frames.push({ t: 0, lm: null });
  // @ts-expect-error Proposed metadata must not become coords-1 fields.
  coords.capture.recorded_at_source = 'metadata';
  // @ts-expect-error A half-null standing interval is not a contract value.
  const marks: StandMarks = { stand_start: 0, stand_end: null };
  // @ts-expect-error A failed detection must carry a reason.
  const detection: StandDetection = { status: 'not_detected',
    marks: { stand_start: null, stand_end: null }, version: 'compile-only' };
  // @ts-expect-error The input requires the source of the recording timestamp.
  const input: ProcessingInput = { subjectId: 'compile-only', heightCm: 170,
    recordedAt: '2026-11-16T10:00:00Z', recordingId: 'compile-only' };
  void marks; void detection; void input;
}
