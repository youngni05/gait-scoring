"""Capture-day corrections for the shared UI. Original packets and reports stay immutable."""
from datetime import date, datetime, time, timedelta, timezone
from uuid import UUID

from fastapi import Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, defer

from app.db import get_db
from app.models import GaitMeasurement, Participant

KST = timezone(timedelta(hours=9))


class CaptureDayEdit(BaseModel):
    day: date
    previous_day: date | None


def capture_day(measurement):
    stamp = measurement.corrected_capture_at or measurement.measured_at
    return stamp.astimezone(KST).date() if stamp else None


def edit_capture_day(participant_id: UUID, recording_id: UUID, body: CaptureDayEdit,
                     db: Session = Depends(get_db)):
    row = db.execute(select(GaitMeasurement, Participant.birth_date)
                     .join(Participant, Participant.id == GaitMeasurement.participant_id)
                     .where(GaitMeasurement.id == recording_id, Participant.id == participant_id)
                     .options(defer(GaitMeasurement.keypoints_json))
                     .with_for_update(of=GaitMeasurement)).one_or_none()
    if row is None:
        raise HTTPException(404, "Recording not found")
    measurement, birth = row
    day = body.day
    if day > datetime.now(KST).date() or (birth and day < birth):
        raise HTTPException(422, "Invalid capture date")
    current = capture_day(measurement)
    if current == day:
        return Response(status_code=204)
    if current != body.previous_day:
        raise HTTPException(409, "Capture date changed")
    measurement.corrected_capture_at = datetime.combine(day, time.min, KST)
    measurement.corrected_capture_age = (day.year - birth.year -
        ((day.month, day.day) < (birth.month, birth.day))) if birth else None
    db.commit()
    return Response(status_code=204)


def corrected_report(report, measurement):
    if not report or not measurement.corrected_capture_at:
        return report
    stamp = measurement.corrected_capture_at.astimezone(KST).isoformat()
    result = {**report, "captured_at": stamp}
    if isinstance(report.get("record"), dict):
        result["record"] = {**report["record"], "captured_at": stamp, "recorded_at_source": "user_entered"}
    return result
