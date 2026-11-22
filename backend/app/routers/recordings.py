"""Authenticated capture submission, shared reports and history for the mobile UI."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, defer

from app.ai_module import LIMITS, RequestError, compare_and_explain, parse_json, to_request
from app.capture_day import capture_day, corrected_report, edit_capture_day
from app.db import get_db
from app.recording_views import legacy_curves, RECENT_VISITS_SQL, SUMMARY_SQL
from app.deps import get_current_user
from app.models import GaitAnalysisResult, GaitMeasurement, Participant, User
from app.routers.measurements import require_participant

router = APIRouter(prefix="/team", tags=["Shared recordings"], dependencies=[Depends(get_current_user)])
router.patch("/participants/{participant_id}/recordings/{recording_id}/capture-day", status_code=204,
             response_class=Response)(edit_capture_day)


async def bounded_json(request: Request):
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(415, "Expected coordinate JSON")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > LIMITS["max_body_bytes"]:
            raise HTTPException(413, "Coordinate payload exceeds limit")
        content.extend(chunk)
    try:
        return parse_json(content)
    except RequestError as e:
        raise HTTPException(e.status, e.code) from None


@router.post("/participants/{participant_id}/recordings", status_code=202)
def submit(participant_id: UUID, response: Response, body=Depends(bounded_json),
           db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if not isinstance(body, dict) or set(body) != {"packet", "fileName"}:
        raise HTTPException(422, "Expected a coordinate packet and file name")
    filename, packet = body["fileName"], body["packet"]
    if not isinstance(filename, str) or not 1 <= len(filename) <= 255:
        raise HTTPException(422, "Invalid file name")
    try:
        to_request(packet)
    except (ValueError, KeyError, TypeError, OverflowError) as e:
        raise HTTPException(422, "Invalid coordinate packet") from e
    coords = packet["coords"]
    recording = UUID(coords["recording_id"])
    if coords["schema_version"] != "team-local-coords/1" or packet["requestId"] != str(recording) or coords["subject"]["id"] != str(participant_id):
        raise HTTPException(422, "Recording or participant mismatch")
    fingerprint = sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    def replay():
        previous = db.scalar(select(GaitMeasurement).where(GaitMeasurement.id == recording)
                             .options(defer(GaitMeasurement.keypoints_json)))
        if previous is None:
            return None
        if (previous.created_by != user.id or previous.participant_id != participant_id
                or (previous.processing_metadata or {}).get("fingerprint") != fingerprint):
            raise HTTPException(409, "Recording ID already used")
        response.status_code = 200
        return {"id": recording, "participant_id": participant_id}

    existing = replay()
    if existing is not None:
        return existing
    person = db.scalar(select(Participant).where(Participant.id == participant_id)
                       .with_for_update(read=True, key_share=True))
    if person is None:
        raise HTTPException(404, "Participant not found")
    stamp = coords["capture"]["recorded_at"]
    received = datetime.now(timezone.utc)
    measured = datetime.fromisoformat(stamp) if stamp else None
    age = None
    if measured:
        day = measured.astimezone(timezone(timedelta(hours=9))).date()
        birth = person.birth_date
        if measured > received or (birth and day < birth):
            raise HTTPException(422, "Invalid capture date")
        if birth:
            age = day.year - birth.year - ((day.month, day.day) < (birth.month, birth.day))
    meta = dict(metadata=packet["metadata"], bridge_context=packet["bridge_context"], file_name=filename,
                height_cm=coords["subject"]["height_cm"], fingerprint=fingerprint)
    inserted = db.scalar(insert(GaitMeasurement).values(
        id=recording, participant_id=participant_id, age=age, measured_at=measured, received_at=received,
        created_by=user.id, keypoints_json=coords, processing_metadata=meta, analysis_status="pending",
        extractor_name="mediapipe_tasks_vision", extractor_version=coords["pose"]["engine_version"],
    ).on_conflict_do_nothing(index_elements=["id"]).returning(GaitMeasurement.id))
    if inserted is None:
        existing = replay()
        if existing is None:
            raise HTTPException(409, "Recording changed during submission")
        return existing
    db.commit()
    return {"id": recording, "participant_id": participant_id}


record_date = func.coalesce(GaitMeasurement.corrected_capture_at, GaitMeasurement.analysis_started_at, GaitMeasurement.measured_at)


@router.get("/participant-summaries")
def summaries(db: Session = Depends(get_db)):
    return [dict(id=p.id, name=p.name, sex={"F": "여", "M": "남"}.get(p.gender), birth=p.birth_date or "",
                 heightCm=p.height_cm and float(p.height_cm), latestMeasuredAt=latest, resultCount=count)
            for p, latest, count in db.execute(select(Participant, func.max(record_date),
                func.count(GaitMeasurement.id)).outerjoin(GaitMeasurement).group_by(Participant.id).order_by(Participant.name, Participant.id))]


def record_query():
    return select(GaitMeasurement, GaitAnalysisResult).options(defer(GaitMeasurement.keypoints_json)).outerjoin(GaitAnalysisResult)


def record_dict(m, result):
    meta = m.processing_metadata or {}
    curves, missing = legacy_curves(result) if m.analysis_status is None else ({}, {})
    return dict(id=m.id, subjectId=m.participant_id, measuredAt=capture_day(m) if m.corrected_capture_at else m.analysis_started_at or m.measured_at or "",
                captureDay=capture_day(m), ageAtCapture=m.corrected_capture_age if m.corrected_capture_at else m.age,
                dateBasis="capture" if m.corrected_capture_at else "analysis_start" if m.analysis_started_at else "legacy_capture" if m.analysis_status is None else "capture",
                heightAtCapture=meta.get("height_cm"), fileName=meta.get("file_name", ""),
                serverState=m.analysis_status or "legacy", live=corrected_report(result.report_json, m) if result else None,
                gdi=result.gdi_score if result else None, cadence=result.cadence_spm if result else None,
                symmetry=None, curves=curves, missing=missing)


@router.get("/participants/{participant_id}/recordings")
def list_records(participant_id: UUID, offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    query = record_query().where(GaitMeasurement.participant_id == participant_id).order_by(
        record_date.desc().nulls_last(), GaitMeasurement.received_at.desc().nulls_last(), GaitMeasurement.id).offset(offset).limit(20)
    return [record_dict(*row) for row in db.execute(query)]


@router.get("/recordings/{recording_id}")
def get_record(recording_id: UUID, db: Session = Depends(get_db)):
    row = db.execute(record_query().where(GaitMeasurement.id == recording_id)).one_or_none()
    if not row:
        raise HTTPException(404, "Measurement not found")
    return record_dict(*row)


@router.delete("/participants/{participant_id}/recordings/{recording_id}", status_code=204)
def delete_record(participant_id: UUID, recording_id: UUID, db: Session = Depends(get_db)):
    m = db.scalar(select(GaitMeasurement).where(GaitMeasurement.id == recording_id,
                  GaitMeasurement.participant_id == participant_id, GaitMeasurement.analysis_status.is_not(None))
                  .options(defer(GaitMeasurement.keypoints_json)).with_for_update())
    if m is None:
        raise HTTPException(404, "Shared recording not found")
    db.execute(delete(GaitAnalysisResult).where(GaitAnalysisResult.measurement_id == m.id))
    db.delete(m)
    db.commit()
    return Response(status_code=204)


@router.post("/recordings/{recording_id}/retry", status_code=202)
def retry_record(recording_id: UUID, db: Session = Depends(get_db)):
    m = db.scalar(select(GaitMeasurement).where(GaitMeasurement.id == recording_id)
                  .options(defer(GaitMeasurement.keypoints_json)).with_for_update())
    if m is None:
        raise HTTPException(404, "Recording not found")
    if m.analysis_status == "failed":
        m.analysis_status = "pending"
    elif m.analysis_status != "pending":
        raise HTTPException(409, "Recording cannot be retried")
    db.commit()
    return {"id": recording_id}


@router.post("/history/explain")
def history(body=Depends(bounded_json)):
    try:
        return compare_and_explain(body)
    except RequestError as e:
        raise HTTPException(e.status, e.code) from None


@router.get("/participant-summaries/{participant_id}")
def participant_summary(participant_id: UUID, db: Session = Depends(get_db)):
    p = require_participant(db, participant_id)
    latest, count = db.execute(select(
        func.max(record_date),
        func.count(GaitMeasurement.id)).where(GaitMeasurement.participant_id == participant_id)).one()
    previews = db.execute(text(RECENT_VISITS_SQL), {"participant_id": participant_id}).scalars().all() if count else []
    return dict(id=p.id, name=p.name, sex={"F": "여", "M": "남"}.get(p.gender), birth=p.birth_date or "",
                heightCm=float(p.height_cm) if p.height_cm is not None else None,
                latestMeasuredAt=latest, resultCount=count, recentVisits=previews)


@router.get("/participants/{participant_id}/recording-summaries")
def recording_summaries(participant_id: UUID, offset: int = Query(0, ge=0),
                        limit: int = Query(20, ge=1, le=20), db: Session = Depends(get_db)):
    require_participant(db, participant_id)
    total = db.scalar(select(func.count(GaitMeasurement.id)).where(GaitMeasurement.participant_id == participant_id))
    rows = db.execute(text(SUMMARY_SQL), dict(participant_id=participant_id, offset=offset, limit=limit)).mappings()
    items = [dict(id=m["id"], subjectId=m["participant_id"], measuredAt=m["corrected_day"] or m["measured_at"] or "",
                  dateBasis=m["date_basis"], ageAtCapture=m["age"], heightAtCapture=m["height_cm"],
                  fileName=m["file_name"] or "", serverState=m["analysis_status"] or "legacy",
                  gdi=m["gdi_score"], cadence=m["cadence_spm"], symmetry=None, curves={}, missing={},
                  preview=m["preview"]) for m in rows]
    return dict(items=items, total=total, offset=offset, limit=limit)
