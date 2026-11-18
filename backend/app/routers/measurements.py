from datetime import timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, defer

from app.db import get_db
from app.deps import get_current_user
from app.models import GaitAnalysisResult, GaitMeasurement, Participant, User
from app.schemas.measurements import (
    MeasurementCreate,
    MeasurementListItem,
    MeasurementResponse,
    MeasurementResultResponse,
)


router = APIRouter(tags=["Measurements"])
KOREA_TIMEZONE = timezone(timedelta(hours=9))


def require_participant(db: Session, participant_id: UUID) -> Participant:
    participant = db.get(Participant, participant_id)
    if participant is None:
        raise HTTPException(status_code=404, detail="Participant not found")
    return participant


@router.post(
    "/participants/{participant_id}/measurements",
    response_model=MeasurementResponse,
    status_code=201,
)
def create_measurement(
    participant_id: UUID,
    data: MeasurementCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    participant = require_participant(db, participant_id)
    if participant.birth_date is None:
        raise HTTPException(status_code=409, detail="Participant birth date is required before recording a measurement")

    capture_date = data.measured_at.astimezone(KOREA_TIMEZONE).date()
    birth_date = participant.birth_date
    if capture_date < birth_date:
        raise HTTPException(status_code=422, detail="Capture date must not precede birth date")
    age = capture_date.year - birth_date.year - (
        (capture_date.month, capture_date.day) < (birth_date.month, birth_date.day)
    )
    measurement = GaitMeasurement(
        participant_id=participant.id,
        age=age,
        created_by=current_user.id,
        **data.model_dump(),
    )
    db.add(measurement)
    db.commit()
    db.refresh(measurement)
    return measurement


@router.get(
    "/participants/{participant_id}/measurements",
    response_model=list[MeasurementListItem],
)
def list_measurements(
    participant_id: UUID,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_participant(db, participant_id)
    rows = db.execute(
        select(GaitMeasurement, GaitAnalysisResult.id)
        .options(defer(GaitMeasurement.keypoints_json, raiseload=True))
        .outerjoin(GaitAnalysisResult, GaitAnalysisResult.measurement_id == GaitMeasurement.id)
        .where(GaitMeasurement.participant_id == participant_id)
        .order_by(GaitMeasurement.measured_at.desc(), GaitMeasurement.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [
        MeasurementListItem(
            **MeasurementResponse.model_validate(measurement).model_dump(),
            has_analysis_result=result_id is not None,
        )
        for measurement, result_id in rows
    ]


@router.get("/measurements/{measurement_id}/result", response_model=MeasurementResultResponse)
def get_measurement_result(
    measurement_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    measurement = db.get(GaitMeasurement, measurement_id)
    if measurement is None:
        raise HTTPException(status_code=404, detail="Measurement not found")
    participant = require_participant(db, measurement.participant_id)
    result = db.scalar(
        select(GaitAnalysisResult).where(GaitAnalysisResult.measurement_id == measurement_id)
    )
    return {
        "status": "available" if result is not None else "not_available",
        "participant": participant,
        "measurement": measurement,
        "analysis_result": result,
    }
