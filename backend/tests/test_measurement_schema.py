import copy
import json
from pathlib import Path
import unittest

from pydantic import ValidationError
from app.schemas.measurements import MeasurementCreate

FIXTURE = Path(__file__).resolve().parents[2] / "contracts/measurement-1/synthetic.request.json"


class MeasurementSchemaTests(unittest.TestCase):
    def setUp(self):
        self.body = json.loads(FIXTURE.read_text(encoding="utf-8"))["body"]

    def test_shared_fixture_and_raw_coordinates_preserved(self):
        original = copy.deepcopy(self.body)
        model = MeasurementCreate.model_validate(self.body)
        self.assertEqual(model.keypoints_json, original["keypoints_json"])
        self.assertEqual(self.body, original)
        self.assertEqual(model.processing_metadata.detection.status, "detected")

    def test_legacy_request_remains_supported(self):
        del self.body["processing_metadata"]
        self.body["keypoints_json"] = {"legacy": [1, 2]}
        self.assertIsNone(MeasurementCreate.model_validate(self.body).processing_metadata)

    def test_invalid_coordinate_shapes_and_values(self):
        for change in (
            lambda c: c.update(video="private.mp4"),
            lambda c: c["frames"].pop(),
            lambda c: c["frames"][0]["lm"].pop(),
            lambda c: c["frames"][0]["lm"][0].pop(),
            lambda c: c["frames"][0]["lm"][0].__setitem__(3, 1.1),
            lambda c: c["frames"][0]["lm"][0].__setitem__(0, float("nan")),
            lambda c: c["frames"][0]["lm"][0].__setitem__(0, True),
            lambda c: c["frames"][1].update(t=c["frames"][0]["t"]),
            lambda c: c["capture"].update(width="1280"),
            lambda c: c["capture"].update(orientation="portrait"),
            lambda c: c["capture"].update(recorded_at="2026-01-01"),
            lambda c: c["capture"].update(device=None),
            lambda c: c["subject"].update(height_cm=99),
            lambda c: c["marks"].update(stand_end=99999),
        ):
            body = copy.deepcopy(self.body)
            change(body["keypoints_json"])
            with self.subTest(change=change), self.assertRaises(ValidationError):
                MeasurementCreate.model_validate(body)

    def test_invalid_metadata_and_cross_fields(self):
        for change in (
            lambda b: b.update(processing_metadata=None),
            lambda b: b.update(extractor_version="wrong"),
            lambda b: b.update(extractor_name="wrong"),
            lambda b: b.update(measured_at="2000-01-01T00:00:00Z"),
            lambda b: b.update(measured_at="2099-01-01T00:00:00Z"),
            lambda b: b["processing_metadata"].update(filePath="private.mp4"),
            lambda b: b["processing_metadata"].update(rotationAppliedDegrees=float("inf")),
            lambda b: b["processing_metadata"].update(modelAssetVersion=" "),
            lambda b: b["processing_metadata"]["detection"].update(status="unknown"),
            lambda b: b["processing_metadata"]["detection"].update(version=" "),
            lambda b: b["processing_metadata"]["detection"]["marks"].update(stand_end=0),
        ):
            body = copy.deepcopy(self.body)
            change(body)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                MeasurementCreate.model_validate(body)

    def test_missing_detection_and_outside_image_xy_are_valid(self):
        c = self.body["keypoints_json"]
        c["marks"] = {"stand_start": None, "stand_end": None}
        self.body["processing_metadata"]["detection"] = {
            "status": "not_detected", "marks": c["marks"], "version": "test", "reason": "occluded",
        }
        c["frames"][0]["lm"][0][0] = -0.25
        c["frames"][0]["lm"][0][1] = 1.2
        model = MeasurementCreate.model_validate(self.body)
        self.assertEqual(model.keypoints_json["frames"][0]["lm"][0][:2], [-0.25, 1.2])


if __name__ == "__main__":
    unittest.main()
