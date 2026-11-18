from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import Participant, User
from app.schemas.participants import (
    ParticipantCreate,
    ParticipantResponse,
    ParticipantUpdate,
)


router = APIRouter(
    prefix="/participants",
    tags=["Participants"],
)


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
