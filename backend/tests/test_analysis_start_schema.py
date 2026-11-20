import copy
import json
from pathlib import Path
import unittest
from pydantic import ValidationError
from app.schemas.measurements import MeasurementCreate


def analysis_body():
    body = json.loads((Path(__file__).resolve().parents[2] / 'contracts/measurement-1/synthetic.request.json').read_text(encoding='utf-8'))['body']
    body.pop('measured_at')
    body['analysis_started_at'] = '2026-01-01T15:00:00Z'
    body['keypoints_json']['schema_version'] = 'coords-2'
    body['keypoints_json']['capture'].pop('recorded_at')
    body['processing_metadata']['schema_version'] = 'processing-2'
    body['processing_metadata'].pop('recordedAtSource')
    return body


class AnalysisStartSchemaTests(unittest.TestCase):
    def test_accepts_start_without_capture_metadata(self):
        data = MeasurementCreate.model_validate(analysis_body())
        self.assertIsNone(data.measured_at)
        self.assertEqual(data.analysis_started_at.isoformat(), '2026-01-01T15:00:00+00:00')
        self.assertNotIn('recorded_at', data.keypoints_json['capture'])

    def test_rejects_missing_start_capture_fields_future_and_mixed_contracts(self):
        for change in [
            lambda b: b.pop('analysis_started_at'),
            lambda b: b.update(analysis_started_at=None),
            lambda b: b.update(analysis_started_at='2099-01-01T00:00:00Z'),
            lambda b: b.update(analysis_started_at='2026-01-01'),
            lambda b: b.update(measured_at=None),
            lambda b: b['keypoints_json']['capture'].update(recorded_at='2020-01-01T00:00:00Z'),
            lambda b: b['processing_metadata'].update(recordedAtSource='metadata'),
            lambda b: b['keypoints_json'].update(schema_version='coords-1'),
        ]:
            body = copy.deepcopy(analysis_body()); change(body)
            with self.assertRaises(ValidationError): MeasurementCreate.model_validate(body)
