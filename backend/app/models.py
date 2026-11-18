from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    login_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(
        String(255), nullable=False
    )


class Participant(Base):
    __tablename__ = "participants"
    __table_args__ = (
        CheckConstraint("gender IN ('M', 'F')", name="ck_participants_gender"),
        CheckConstraint("height_cm > 0", name="ck_participants_height_cm"),
        Index("ix_participants_created_by", "created_by"),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    created_by: Mapped[UUID] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    gender: Mapped[str | None] = mapped_column(String(1), nullable=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    height_cm: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)


class GaitMeasurement(Base):
    __tablename__ = "gait_measurements"
    __table_args__ = (
        CheckConstraint("age >= 0", name="ck_gait_measurements_age"),
        Index("ix_gait_measurements_participant_measured_id", "participant_id", "measured_at", "id"),
        Index("ix_gait_measurements_created_by", "created_by"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    participant_id: Mapped[UUID] = mapped_column(ForeignKey("participants.id"), nullable=False)
    # Completed age at capture time; populated by the measurement service.
    age: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    keypoints_json: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=False)
    processing_metadata: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    extractor_name: Mapped[str] = mapped_column(String(100), nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(100), nullable=False)
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)


class MeasurementRequest(Base):
    __tablename__ = "measurement_requests"

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), primary_key=True)
    request_key: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    participant_id: Mapped[UUID] = mapped_column(ForeignKey("participants.id"), nullable=False)
    body_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    measurement_id: Mapped[UUID] = mapped_column(ForeignKey("gait_measurements.id"), nullable=False)


class GaitAnalysisResult(Base):
    __tablename__ = "gait_analysis_results"
    __table_args__ = (
        UniqueConstraint("measurement_id", name="uq_gait_analysis_results_measurement_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    measurement_id: Mapped[UUID] = mapped_column(ForeignKey("gait_measurements.id"), nullable=False)
    gdi_score: Mapped[float | None] = mapped_column(Double, nullable=True)
    stride_length_cm: Mapped[float | None] = mapped_column(Double, nullable=True)
    cadence_spm: Mapped[float | None] = mapped_column(Double, nullable=True)
    hip_graph_json: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    knee_graph_json: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    ankle_graph_json: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
