import json
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from app.schemas.participants import ParticipantResponse
from app.schemas.coords import Coords, ProcessingMetadata


class MeasurementCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keypoints_json: dict[str, JsonValue] | list[JsonValue]
    extractor_name: str = Field(min_length=1, max_length=100)
    extractor_version: str = Field(min_length=1, max_length=100)
    measured_at: AwareDatetime
    processing_metadata: ProcessingMetadata | None = None

    @model_validator(mode="after")
    def validate_contract(self):
        if "processing_metadata" in self.model_fields_set and self.processing_metadata is None:
            raise ValueError("processing_metadata must not be null when supplied")
        if self.processing_metadata is not None:
            coords = Coords.model_validate(self.keypoints_json)
            if self.extractor_name != "mediapipe_tasks_vision" or self.extractor_version != coords.pose.engine_version:
                raise ValueError("Extractor metadata does not match coords-1")
            captured = datetime.fromisoformat(coords.capture.recorded_at.replace("Z", "+00:00"))
            if captured != self.measured_at:
                raise ValueError("Capture timestamp does not match measured_at")
            if self.processing_metadata.detection.marks.model_dump() != coords.marks.model_dump():
                raise ValueError("Detection marks do not match coords-1")
        return self

    @field_validator("extractor_name", "extractor_version", mode="before")
    @classmethod
    def strip_extractor_metadata(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("keypoints_json")
    @classmethod
    def validate_keypoints(cls, value):
        if not value:
            raise ValueError("Keypoints must be a non-empty JSON object or array")
        # JSONB cannot store NaN or Infinity, including in legacy requests.
        json.dumps(value, allow_nan=False)
        return value

    @field_validator("measured_at")
    @classmethod
    def validate_capture_time(cls, value: datetime) -> datetime:
        if value > datetime.now(timezone.utc):
            raise ValueError("Capture time must not be in the future")
        return value.astimezone(timezone.utc)


class MeasurementResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    participant_id: UUID
    age: int
    extractor_name: str
    extractor_version: str
    measured_at: datetime
    created_by: UUID


class MeasurementListItem(MeasurementResponse):
    has_analysis_result: bool


class AnalysisResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    measurement_id: UUID
    gdi_score: float | None
    stride_length_cm: float | None
    cadence_spm: float | None
    hip_graph_json: JsonValue
    knee_graph_json: JsonValue
    ankle_graph_json: JsonValue
    model_version: str
    analyzed_at: datetime


class MeasurementResultResponse(BaseModel):
    status: Literal["available", "not_available"]
    participant: ParticipantResponse
    measurement: MeasurementResponse
    analysis_result: AnalysisResultResponse | None
