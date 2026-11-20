import hashlib
import re
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from sqlalchemy import select, func
from sqlalchemy.orm import Session, defer

from app.db import get_db
from app.deps import get_current_user
from app.models import GaitAnalysisResult, GaitMeasurement, Participant, User
from app.measurement_service import store_measurement, problem
from app.schemas.measurements import (
    MeasurementCreate,
    MeasurementListItem,
    MeasurementResponse,
    MeasurementResultResponse,
)


router = APIRouter(tags=["Measurements"])


async def request_identity(
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> tuple[UUID | None, str]:
    keys = request.headers.getlist("Idempotency-Key")
    if not keys:
        return None, ""
    if len(keys) != 1 or not re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", keys[0]):
        raise problem("INVALID_REQUEST_KEY", "Idempotency-Key must be one UUID", 422)
    return UUID(keys[0]), hashlib.sha256(await request.body()).hexdigest()


def require_participant(db: Session, participant_id: UUID) -> Participant:
    participant = db.get(Participant, participant_id)
    if participant is None:
        raise HTTPException(status_code=404, detail="Participant not found")
    return participant


@router.post(
    "/participants/{participant_id}/measurements",
    response_model=MeasurementResponse,
    status_code=201,
    responses={200: {"model": MeasurementResponse, "description": "Previously accepted identical request"},
               409: {"description": "Missing birth date, request conflict, or retryable lock wait"}},
)
def create_measurement(
    participant_id: UUID,
    data: MeasurementCreate,
    response: Response,
    identity: tuple[UUID | None, str] = Depends(request_identity),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    measurement, created = store_measurement(
        db, current_user.id, participant_id, data, identity[0], identity[1],
    )
    response.status_code = 201 if created else 200
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
        .options(defer(GaitMeasurement.keypoints_json, raiseload=True),
                 defer(GaitMeasurement.processing_metadata, raiseload=True))
        .outerjoin(GaitAnalysisResult, GaitAnalysisResult.measurement_id == GaitMeasurement.id)
        .where(GaitMeasurement.participant_id == participant_id)
        .order_by(func.coalesce(GaitMeasurement.analysis_started_at, GaitMeasurement.measured_at).desc(), GaitMeasurement.id.desc())
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
