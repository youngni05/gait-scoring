"""Atomic measurement creation and durable, per-user idempotency."""
from datetime import timedelta, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.models import GaitMeasurement, MeasurementRequest, Participant
from app.schemas.measurements import MeasurementCreate


def problem(code: str, message: str, status: int = 409, headers=None):
    return HTTPException(status_code=status, detail={"code": code, "message": message}, headers=headers)


def replay(db: Session, previous: MeasurementRequest, participant_id: UUID, fingerprint: str):
    if previous.participant_id != participant_id or previous.body_sha256 != fingerprint:
        raise problem("IDEMPOTENCY_CONFLICT", "The request key is already associated with different data")
    return db.get(GaitMeasurement, previous.measurement_id), False


def store_measurement(db: Session, user_id: UUID, participant_id: UUID,
                      data: MeasurementCreate, request_key: UUID | None, fingerprint: str):
    modern = data.processing_metadata is not None
    if modern != (request_key is not None):
        raise problem("INVALID_MEASUREMENT_CONTRACT", "Metadata and Idempotency-Key must be supplied together", 422)
    if modern and data.keypoints_json["subject"]["id"] != str(participant_id):
        raise problem("PARTICIPANT_MISMATCH", "Coordinates must match the selected participant", 422)
    participant = db.get(Participant, participant_id)
    if participant is None:
        raise HTTPException(status_code=404, detail="Participant not found")
    if modern:
        previous = db.get(MeasurementRequest, (user_id, request_key))
        if previous is not None:
            return replay(db, previous, participant_id, fingerprint)
    # Replay precedes mutable profile checks: a birth-date correction must not
    # change the first response or make a committed request impossible to retry.
    if participant.birth_date is None:
        if modern:
            raise problem("PARTICIPANT_BIRTH_DATE_REQUIRED", "Birth date is required before recording a measurement")
        raise HTTPException(status_code=409, detail="Participant birth date is required before recording a measurement")
    reference_time = data.analysis_started_at or data.measured_at
    reference_date = reference_time.astimezone(timezone(timedelta(hours=9))).date()
    birth_date = participant.birth_date
    if reference_date < birth_date:
        raise HTTPException(status_code=422, detail="Reference date must not precede birth date")
    age = reference_date.year - birth_date.year - ((reference_date.month, reference_date.day) < (birth_date.month, birth_date.day))
    values = data.model_dump(exclude={"processing_metadata"})
    metadata = data.processing_metadata.model_dump(exclude_unset=True) if modern else None
    measurement = GaitMeasurement(participant_id=participant_id, age=age, created_by=user_id,
                                  processing_metadata=metadata, **values)
    try:
        if modern:
            db.execute(text("SET LOCAL lock_timeout = '2s'"))
        db.add(measurement)
        db.flush()
        if modern:
            inserted = db.scalar(
                insert(MeasurementRequest).values(
                    user_id=user_id, request_key=request_key, participant_id=participant_id,
                    body_sha256=fingerprint, measurement_id=measurement.id,
                ).on_conflict_do_nothing(index_elements=["user_id", "request_key"])
                .returning(MeasurementRequest.request_key)
            )
            if inserted is None:
                # Undo the losing measurement too, then read the winning transaction.
                db.rollback()
                previous = db.get(MeasurementRequest, (user_id, request_key))
                if previous is None:
                    raise problem("REQUEST_IN_PROGRESS", "Retry the same request shortly", headers={"Retry-After": "2"})
                return replay(db, previous, participant_id, fingerprint)
        db.commit()
        db.refresh(measurement)
        return measurement, True
    except DBAPIError as error:
        db.rollback()
        if modern and getattr(error.orig, "sqlstate", None) == "55P03":
            raise problem("REQUEST_IN_PROGRESS", "Retry the same request shortly", headers={"Retry-After": "2"}) from error
        raise
