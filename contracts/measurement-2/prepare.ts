import { prepareMeasurement, type PreparedMeasurement } from '../measurement-1/prepare';
import type { UploadRequest } from '../../mobile/src/pose/upload-gateway';

/** Reuse the local pose validator, then emit the versioned, capture-time-free contract. */
export function prepareAnalysisMeasurement(request: UploadRequest, participantId: string,
  analysisStartedAt: string, now: Date = new Date()): PreparedMeasurement {
  const started = new Date(analysisStartedAt);
  if (!/^\d{4}-\d{2}-\d{2}T.*(Z|[+-]\d{2}:\d{2})$/.test(analysisStartedAt) ||
      !Number.isFinite(started.getTime()) || started.getTime() > now.getTime()) {
    throw new Error('분석 시작 시각이 올바르지 않습니다. 기기의 날짜와 시간을 확인하세요.');
  }
  // Legacy serializer is an internal adapter only: neither its timestamp nor its source is sent.
  const local = { ...request, coords: { ...request.coords,
    capture: { ...request.coords.capture, recorded_at: started.toISOString() } } };
  const prepared = prepareMeasurement(local, participantId, now);
  const { recorded_at: _captureTime, ...capture } = local.coords.capture;
  const { recordedAtSource: _source, ...metadata } = request.metadata;
  return Object.freeze({ ...prepared, body: JSON.stringify({
    keypoints_json: { ...request.coords, schema_version: 'coords-2', capture },
    extractor_name: 'mediapipe_tasks_vision', extractor_version: request.coords.pose.engine_version,
    analysis_started_at: started.toISOString(),
    processing_metadata: { schema_version: 'processing-2', ...metadata },
  }) });
}
