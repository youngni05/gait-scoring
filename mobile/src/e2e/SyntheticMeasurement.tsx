import { useState } from 'react';
import fixture from '../../../contracts/measurement-1/synthetic.request.json';
import { MeasurementUpload } from '../MeasurementUpload';
import { validateUploadRequest, type UploadRequest } from '../pose/upload-gateway';

/** Loaded only by the explicit DEV E2E build. Never an alternate production input. */
export default function SyntheticMeasurement({ participantId }: { participantId: string }) {
  const [request, setRequest] = useState<UploadRequest | null>(null);
  function prepare() {
    const body = structuredClone(fixture.body);
    body.keypoints_json.subject.id = participantId;
    body.keypoints_json.recording_id = crypto.randomUUID();
    const { schema_version: _, ...metadata } = body.processing_metadata;
    const value = { coords: body.keypoints_json, metadata, requestId: crypto.randomUUID() } as UploadRequest;
    validateUploadRequest(value); setRequest(value);
  }
  return <section aria-label="합성 좌표 자동 테스트"><p>자동 테스트 전용 합성 좌표입니다. 실제 영상 추출 검증이 아닙니다.</p>
    <button disabled={!!request} onClick={prepare}>합성 좌표 준비</button>
    {request && <MeasurementUpload request={request} />}
  </section>;
}
