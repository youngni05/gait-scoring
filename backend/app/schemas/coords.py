"""Server validation for coords-1. Validate without replacing the stored raw JSON."""
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class StrictObject(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


NonBlank = Annotated[str, Field(min_length=1, pattern=r"\S")]
Version = Annotated[str, Field(min_length=1, max_length=100, pattern=r"\S")]
Nonnegative = Annotated[FiniteFloat, Field(ge=0)]


class Subject(StrictObject):
    id: NonBlank
    height_cm: Annotated[FiniteFloat, Field(ge=100, le=220)]


class Capture(StrictObject):
    mode: Literal["guided", "fixed"]
    recorded_at: Annotated[str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$")]
    width: Annotated[int, Field(ge=1)]
    height: Annotated[int, Field(ge=1)]
    orientation: Literal["landscape", "portrait"]
    device: NonBlank = Field(default=None)  # optional, but explicit null is invalid

    @model_validator(mode="after")
    def validate_capture(self):
        datetime.fromisoformat(self.recorded_at.replace("Z", "+00:00"))
        if (self.width > self.height and self.orientation != "landscape") or (
            self.height > self.width and self.orientation != "portrait"
        ):
            raise ValueError("Orientation must match rotated dimensions")
        return self


class Pose(StrictObject):
    model: Literal["pose_landmarker_lite", "pose_landmarker_full", "pose_landmarker_heavy"]
    engine_version: NonBlank
    delegate: Literal["CPU", "GPU"] = Field(default=None)


class NumericMarks(StrictObject):
    stand_start: Nonnegative
    stand_end: Nonnegative


class MissingMarks(StrictObject):
    stand_start: None
    stand_end: None


Marks = NumericMarks | MissingMarks
Landmark = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]


class Frame(StrictObject):
    t: Nonnegative
    lm: Annotated[list[Landmark], Field(min_length=33, max_length=33)] | None

    @model_validator(mode="after")
    def validate_visibility(self):
        if self.lm is not None and any(not 0 <= point[3] <= 1 for point in self.lm):
            raise ValueError("Visibility must be between 0 and 1")
        return self


class Coords(StrictObject):
    schema_version: Literal["coords-1"]
    recording_id: Annotated[str, Field(pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")]
    subject: Subject
    capture: Capture
    pose: Pose
    marks: Marks
    frames: Annotated[list[Frame], Field(min_length=60)]

    @model_validator(mode="after")
    def validate_sequence(self):
        UUID(self.recording_id)
        if any(b.t <= a.t for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("Frame timestamps must strictly increase")
        if isinstance(self.marks, NumericMarks) and not (
            self.marks.stand_start < self.marks.stand_end <= self.frames[-1].t
        ):
            raise ValueError("Standing interval must lie within observed time coverage")
        return self


class Detected(StrictObject):
    status: Literal["detected"]
    marks: NumericMarks
    version: Version


class NotDetected(StrictObject):
    status: Literal["not_detected"]
    marks: MissingMarks
    version: Version
    reason: NonBlank


class ProcessingMetadata(StrictObject):
    schema_version: Literal["processing-1"]
    recordedAtSource: Literal["metadata", "user_entered"]
    rotationAppliedDegrees: FiniteFloat | None
    detection: Annotated[Detected | NotDetected, Field(discriminator="status")]
    modelAssetVersion: NonBlank = Field(default=None)
