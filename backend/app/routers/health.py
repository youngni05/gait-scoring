from fastapi import APIRouter, HTTPException
from functools import lru_cache
from pathlib import Path
import os
from fastapi.responses import FileResponse
from app.ai_module import POSE_MODEL_SHA256, verify
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db import engine

router = APIRouter(tags=["Health"])


@lru_cache(maxsize=1)
def verified_pose_model():
    path = Path(os.environ.get("GAIT_POSE_MODEL", "/models/pose_landmarker_full.task"))
    verify(path, POSE_MODEL_SHA256)
    return path


@router.get("/assets/pose-full.task", include_in_schema=False)
def pose_model():
    try:
        path = verified_pose_model()
    except (OSError, ValueError):
        raise HTTPException(503, "Verified pose model unavailable") from None
    return FileResponse(path, media_type="application/octet-stream")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/db-health")
def db_health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return {"database": "connected"}

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )
