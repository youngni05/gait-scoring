from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select, func, delete
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import GaitAnalysisResult, GaitMeasurement, MeasurementRequest, Participant, User
from app.schemas.participants import (
    ParticipantCreate,
    ParticipantResponse,
    ParticipantSummary,
    ParticipantUpdate,
)


router = APIRouter(
    prefix="/participants",
    tags=["Participants"],
)


@router.delete("/{participant_id}", status_code=204)
def delete_participant(
    participant_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Team-shared access, like update. Lock before deleting any dependent rows;
    # an active upload/worker must finish before the user can retry deletion.
    try:
        participant = db.scalar(select(Participant).where(Participant.id == participant_id)
                                .with_for_update(nowait=True))
        if participant is None:
            return Response(status_code=204)
        rows = db.execute(select(GaitMeasurement.id, GaitMeasurement.analysis_status)
                          .where(GaitMeasurement.participant_id == participant_id)
                          .with_for_update(nowait=True)).all()
        if any(status == "pending" for _, status in rows):
            raise HTTPException(409, "Analysis is pending; retry after it finishes")
        ids = [mid for mid, _ in rows]
        db.execute(delete(MeasurementRequest).where(MeasurementRequest.participant_id == participant_id))
        db.execute(delete(GaitAnalysisResult).where(GaitAnalysisResult.measurement_id.in_(ids)))
        db.execute(delete(GaitMeasurement).where(GaitMeasurement.participant_id == participant_id))
        db.delete(participant)
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except DBAPIError as exc:
        db.rollback()
        if getattr(exc.orig, "sqlstate", None) == "55P03":
            raise HTTPException(409, "Participant is busy; retry after processing finishes") from None
        raise
    return Response(status_code=204)


@router.post(
    "",
    response_model=ParticipantResponse,
    status_code=201,
)
def create_participant(
    data: ParticipantCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    participant = Participant(
        name=data.name,
        gender=data.gender,
        birth_date=data.birth_date,
        height_cm=data.height_cm,
        created_by=current_user.id,
    )

    db.add(participant)
    db.commit()
    db.refresh(participant)

    return participant


@router.patch("/{participant_id}", response_model=ParticipantResponse)
def update_participant(
    participant_id: UUID,
    data: ParticipantUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    participant = db.get(Participant, participant_id)
    if participant is None:
        raise HTTPException(status_code=404, detail="Participant not found")

    # Only supplied fields change; creator and historical measurement ages stay intact.
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(participant, field, value)
    db.commit()
    db.refresh(participant)
    return participant


@router.get(
    "",
    response_model=list[ParticipantResponse],
)
def get_participants(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.scalars(select(Participant)).all()



def participant_summary_query():
    counts = (
        select(
            GaitMeasurement.participant_id.label("participant_id"),
            func.count(GaitMeasurement.id).label("measurement_count"),
            func.max(func.coalesce(GaitMeasurement.analysis_started_at, GaitMeasurement.measured_at)).label("latest_measured_at"),
        )
        .group_by(GaitMeasurement.participant_id)
        .subquery()
    )
    return (
        select(Participant, func.coalesce(counts.c.measurement_count, 0), counts.c.latest_measured_at)
        .outerjoin(counts, counts.c.participant_id == Participant.id)
    )


def summary_response(row) -> ParticipantSummary:
    participant, count, latest = row
    return ParticipantSummary(
        **ParticipantResponse.model_validate(participant).model_dump(),
        measurement_count=count, latest_measured_at=latest,
    )


# Static route must precede /{participant_id}.
@router.get("/summaries", response_model=list[ParticipantSummary])
def list_participant_summaries(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = db.execute(participant_summary_query().order_by(Participant.name, Participant.id)).all()
    return [summary_response(row) for row in rows]


@router.get("/{participant_id}/summary", response_model=ParticipantSummary)
def get_participant_summary(
    participant_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.execute(participant_summary_query().where(Participant.id == participant_id)).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Participant not found")
    return summary_response(row)

@router.get(
    "/{participant_id}",
    response_model=ParticipantResponse,
)
def get_participant(
    participant_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    participant = db.get(Participant, participant_id)

    if participant is None:
        raise HTTPException(
            status_code=404,
            detail="Participant not found",
        )

    return participant
