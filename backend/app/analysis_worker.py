"""PostgreSQL owns the pending work even after the submitting phone disconnects."""
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.ai_module import analyze_coords
from app.db import SessionLocal
from app.models import GaitAnalysisResult, GaitMeasurement

log = logging.getLogger(__name__)


def process_one(db):
    # A terminated worker releases its lock. The next worker sees the same pending row.
    with db.begin():
        m = db.scalar(select(GaitMeasurement).where(GaitMeasurement.analysis_status == "pending")
                      .order_by(GaitMeasurement.received_at, GaitMeasurement.id)
                      .limit(1).with_for_update(skip_locked=True))
        if m is None:
            return False
        meta = m.processing_metadata
        packet = dict(requestId=str(m.id), coords=m.keypoints_json, metadata=meta["metadata"],
                      bridge_context=meta["bridge_context"])
        try:
            report = analyze_coords(packet)
        except Exception:
            log.exception("Analysis failed for recording %s", m.id)
            m.analysis_status = "failed"
            return True
        now = datetime.now(timezone.utc)
        report.update(id=str(m.id), subject_id=str(m.participant_id), capture_mode=m.keypoints_json["capture"]["mode"],
                      captured_at=report["record"]["captured_at"], saved_at=now.isoformat(), fileName=meta["file_name"])
        db.add(GaitAnalysisResult(measurement_id=m.id, report_json=report, analyzed_at=now,
                                  model_version=report["analysis"]["analysis_version"]))
        m.analysis_status = "ready"
    return True


def main():
    from gdi_adapter import configure
    configure(Path(os.environ["GAIT_MODEL_ASSETS"]), Path(os.environ["GAIT_STT_SOURCE"]))
    log.info("Analysis worker ready")
    while True:
        try:
            with SessionLocal() as db:
                if process_one(db):
                    continue
        except OperationalError:
            log.exception("Database unavailable")
        time.sleep(2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
