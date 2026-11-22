from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ParticipantCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    gender: Literal["M", "F"]
    birth_date: date
    height_cm: Decimal = Field(gt=0, max_digits=5, decimal_places=2)

    @field_validator("birth_date")
    @classmethod
    def reject_future_birth_date(cls, value: date) -> date:
        today_in_korea = datetime.now(timezone(timedelta(hours=9))).date()
        if value > today_in_korea:
            raise ValueError("Birth date must not be in the future")
        return value


class ParticipantUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=100)
    gender: Literal["M", "F"] | None = None
    birth_date: date | None = None
    height_cm: Decimal | None = Field(default=None, gt=0, max_digits=5, decimal_places=2)

    @field_validator("birth_date")
    @classmethod
    def reject_future_birth_date(cls, value: date | None) -> date | None:
        if value is not None:
            return ParticipantCreate.reject_future_birth_date(value)
        return value

    @model_validator(mode="after")
    def require_non_null_changes(self):
        if not self.model_fields_set:
            raise ValueError("At least one participant field must be provided")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("Provided participant fields must not be null")
        return self


class ParticipantResponse(BaseModel):
    id: UUID
    name: str
    gender: Literal["M", "F"] | None
    birth_date: date | None
    height_cm: Decimal | None
    created_by: UUID
    created_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class ParticipantSummary(ParticipantResponse):
    latest_measured_at: datetime | None
    measurement_count: int
