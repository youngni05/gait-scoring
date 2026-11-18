import { validateUploadRequest, type UploadRequest } from '../../mobile/src/pose/upload-gateway';

/** Proposed wire contract. Backend support must land before enabling HTTP transport. */
export interface PreparedMeasurement {
  readonly method: 'POST';
  readonly path: string;
  readonly headers: Readonly<{
    'Content-Type': 'application/json';
    'Idempotency-Key': string;
  }>;
  /** Serialized once: reuse these exact bytes when retrying. No token or video. */
  readonly body: string;
}

const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function prepareMeasurement(
  request: UploadRequest,
  participantId: string,
  now: Date = new Date(),
): PreparedMeasurement {
  validateUploadRequest(request);
  if (!uuid.test(participantId) || participantId !== request.coords.subject.id) {
    throw new Error('Measurement must match the selected participant UUID');
  }
  if (!uuid.test(request.requestId)) throw new Error('Request ID must be a UUID');
  const captured = Date.parse(request.coords.capture.recorded_at);
  if (!Number.isFinite(now.getTime()) || !Number.isFinite(captured) || captured > now.getTime()) {
    throw new Error('Capture time must be valid and not in the future');
  }
  const metadata = request.metadata;
  if (!metadata.detection.version.trim() || metadata.detection.version.length > 100) {
    throw new Error('Detection version must contain 1 to 100 characters');
  }
  if (metadata.detection.status === 'not_detected' && !metadata.detection.reason.trim()) {
    throw new Error('Missing detection requires a reason');
  }
  if (metadata.modelAssetVersion !== undefined && !metadata.modelAssetVersion.trim()) {
    throw new Error('Model asset version must not be blank');
  }
  const extractorVersion = request.coords.pose.engine_version;
  if (!extractorVersion.trim() || extractorVersion.length > 100) {
    throw new Error('Extractor version must contain 1 to 100 characters');
  }
  const body = JSON.stringify({
    keypoints_json: request.coords,
    extractor_name: 'mediapipe_tasks_vision',
    extractor_version: extractorVersion,
    measured_at: request.coords.capture.recorded_at,
    processing_metadata: {
      schema_version: 'processing-1',
      ...metadata,
    },
  });
  return Object.freeze({
    method: 'POST',
    path: `/participants/${participantId}/measurements`,
    headers: Object.freeze({ 'Content-Type': 'application/json', 'Idempotency-Key': request.requestId }),
    body,
  });
}
