"""Real PostgreSQL + real HTTP tests; never use an application database.

Requires GAIT_TEST_DB_WRITES=1 and a DATABASE_URL database name ending in _ci.
Uses one generated, isolated schema and removes only that schema afterwards.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

os.environ.setdefault("JWT_SECRET_KEY", secrets.token_urlsafe(32))

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select, func, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from app.models import GaitAnalysisResult, GaitMeasurement, MeasurementRequest, Participant, User
from app.security import create_access_token, hash_password

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT.parent / "contracts/measurement-1/synthetic.request.json"


class MeasurementPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv("GAIT_TEST_DB_WRITES") != "1":
            raise RuntimeError("Set GAIT_TEST_DB_WRITES=1 only for a disposable _ci database")
        url = make_url(os.environ["DATABASE_URL"])
        if not url.drivername.startswith("postgresql") or not (url.database or "").endswith("_ci"):
            raise RuntimeError("Integration tests require a disposable PostgreSQL database ending in _ci")
        cls.schema = "measurement_test_" + uuid4().hex
        cls.admin = create_engine(url)
        with cls.admin.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{cls.schema}"'))
        cls.addClassCleanup(cls.clean_schema)
        cls.url = url.update_query_dict({"options": f"-csearch_path={cls.schema}"}).render_as_string(hide_password=False)
        cls.engine = create_engine(cls.url)
        cls.addClassCleanup(cls.engine.dispose)
        cfg = Config(str(ROOT / "alembic.ini"))
        original_url = os.environ["DATABASE_URL"]
        try:
            os.environ["DATABASE_URL"] = cls.url
            command.upgrade(cfg, "8b73d4a1c902")
            cls.legacy_user, cls.legacy_participant, cls.legacy_measurement = uuid4(), uuid4(), uuid4()
            with cls.engine.begin() as conn:
                conn.execute(text("INSERT INTO users (id,login_id,password_hash) VALUES (:id,'migration_fixture','unused')"), {"id": cls.legacy_user})
                conn.execute(text("INSERT INTO participants (id,created_by,name) VALUES (:id,:uid,'Migration fixture')"), {"id": cls.legacy_participant, "uid": cls.legacy_user})
                conn.execute(text("""INSERT INTO gait_measurements
                    (id,participant_id,age,keypoints_json,extractor_name,extractor_version,measured_at,created_by)
                    VALUES (:id,:pid,75,'{}'::jsonb,'legacy','1','2026-01-01T00:00:00Z',:uid)"""),
                    {"id": cls.legacy_measurement, "pid": cls.legacy_participant, "uid": cls.legacy_user})
            command.upgrade(cfg, "head")
            command.check(cfg)
        finally:
            os.environ["DATABASE_URL"] = original_url
        cls.user_id, cls.other_user_id = uuid4(), uuid4()
        cls.password = secrets.token_urlsafe(20)
        cls.login_id = "test-" + uuid4().hex
        with Session(cls.engine) as db:
            db.add_all([
                User(id=cls.user_id, login_id=cls.login_id, password_hash=hash_password(cls.password)),
                User(id=cls.other_user_id, login_id="other-" + uuid4().hex, password_hash="unused"),
            ])
            db.commit()
        cls.token = create_access_token(cls.user_id)
        cls.other_token = create_access_token(cls.other_user_id)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.base = f"http://127.0.0.1:{port}"
        cls.log = tempfile.TemporaryFile()
        cls.addClassCleanup(cls.log.close)
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env={**os.environ, "DATABASE_URL": cls.url, "CORS_ALLOW_ORIGINS": "http://localhost:5173"},
            stdout=cls.log, stderr=subprocess.STDOUT,
        )
        cls.addClassCleanup(cls.stop_server)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if cls.server.poll() is not None:
                cls.log.seek(0)
                raise RuntimeError(cls.log.read().decode(errors="replace"))
            try:
                with urlopen(cls.base + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (URLError, TimeoutError):
                time.sleep(0.1)
        raise RuntimeError("API did not become ready")

    @classmethod
    def stop_server(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.server.kill()
            cls.server.wait(timeout=5)

    @classmethod
    def clean_schema(cls):
        # Generated fixed-prefix UUID name; no user-provided schema identifier.
        with cls.admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{cls.schema}" CASCADE'))
        cls.admin.dispose()

    def setUp(self):
        self.participant_id = uuid4()
        with Session(self.engine) as db:
            db.add(Participant(id=self.participant_id, created_by=self.user_id, name="Synthetic participant",
                               gender="F", birth_date=date(1950, 1, 1), height_cm=170))
            db.commit()
        self.body = json.loads(FIXTURE.read_text(encoding="utf-8"))["body"]
        self.body["keypoints_json"]["subject"]["id"] = str(self.participant_id)
        self.key = str(uuid4())
        self.path = f"/participants/{self.participant_id}/measurements"

    def http(self, path, method="GET", body=None, key=None, token="default", headers=None):
        h = {"Authorization": "Bearer " + (self.token if token == "default" else token)} if token else {}
        if key is not None:
            h["Idempotency-Key"] = key
        if body is not None:
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        raw = json.dumps(body, separators=(",", ":")).encode() if isinstance(body, dict) else body
        try:
            response = urlopen(Request(self.base + path, method=method, headers=h, data=raw), timeout=12)
        except HTTPError as error:
            response = error
        with response:
            data = response.read()
            try:
                data = json.loads(data)
            except ValueError:
                pass
            return response.status, data, response.headers

    def count(self):
        with Session(self.engine) as db:
            return db.scalar(select(func.count()).select_from(GaitMeasurement).where(GaitMeasurement.participant_id == self.participant_id))

    def test_delete_participant_cascades_only_target_and_preserves_summary(self):
        status, measurement, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual(status, 201)
        mid = UUID(measurement["id"])
        with Session(self.engine) as db:
            db.add(GaitAnalysisResult(measurement_id=mid, model_version="synthetic",
                                      analyzed_at=datetime.now(timezone.utc)))
            db.commit()
        # Registering a different participant must not change the existing summary.
        summary_path = f"/team/participant-summaries/{self.participant_id}"
        before = self.http(summary_path)[1]
        status, other, _ = self.http("/participants", "POST", {
            "name": "Deletion isolation fixture", "gender": "F", "birth_date": "1950-01-01", "height_cm": 160,
        })
        self.assertEqual(status, 201)
        self.assertEqual(self.http(summary_path)[1], before)
        endpoint = f"/participants/{self.participant_id}"
        self.assertEqual(self.http(endpoint, "DELETE", token=None)[0], 401)
        self.assertEqual(self.http(endpoint, "DELETE", token=self.other_token)[0], 204)
        self.assertEqual(self.http(endpoint, "DELETE")[0], 204)
        self.assertEqual(self.http(endpoint)[0], 404)
        self.assertEqual(self.http(self.path, "POST", self.body, self.key)[0], 404)
        self.assertEqual(self.http(f'/participants/{other["id"]}')[0], 200)
        self.assertEqual(self.http(f'/participants/{self.legacy_participant}')[0], 200)
        with Session(self.engine) as db:
            self.assertIsNone(db.get(GaitMeasurement, mid))
            self.assertIsNone(db.scalar(select(GaitAnalysisResult).where(GaitAnalysisResult.measurement_id == mid)))
            self.assertIsNone(db.get(MeasurementRequest, (self.user_id, UUID(self.key))))

    def test_delete_participant_rejects_pending_and_active_worker(self):
        status, measurement, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual(status, 201)
        mid = UUID(measurement["id"])
        endpoint = f"/participants/{self.participant_id}"
        with Session(self.engine) as db:
            db.get(GaitMeasurement, mid).analysis_status = "pending"
            db.commit()
        self.assertEqual(self.http(endpoint, "DELETE")[0], 409)
        self.assertEqual(self.count(), 1)
        with Session(self.engine) as db:
            db.get(GaitMeasurement, mid).analysis_status = "ready"
            db.commit()
            db.scalar(select(GaitMeasurement).where(GaitMeasurement.id == mid).with_for_update())
            self.assertEqual(self.http(endpoint, "DELETE")[0], 409)
        with Session(self.engine) as db:
            db.scalar(select(Participant).where(Participant.id == self.participant_id)
                      .with_for_update(read=True, key_share=True))
            self.assertEqual(self.http(endpoint, "DELETE")[0], 409)
        self.assertEqual(self.http(endpoint)[0], 200)
        self.assertEqual(self.http(endpoint, "DELETE")[0], 204)

    def test_edit_capture_day_preserves_original_and_reorders_visits(self):
        from app.models import GaitAnalysisResult
        first, second = uuid4(), uuid4()
        stamp = datetime(2025, 7, 1, tzinfo=timezone.utc)
        report = {"captured_at": None, "record": {"captured_at": None}, "analysis": {"segments": []}}
        path = f"/team/participants/{self.participant_id}/recordings"
        with Session(self.engine) as db:
            for mid, measured in [(first, None), (second, stamp)]:
                db.add(GaitMeasurement(id=mid, participant_id=self.participant_id, created_by=self.user_id,
                    keypoints_json={}, extractor_name="test", extractor_version="1", measured_at=measured,
                    processing_metadata={"height_cm": 166.5}, analysis_status="ready"))
            db.flush()
            db.add(GaitAnalysisResult(measurement_id=first, report_json=report, model_version="test", analyzed_at=stamp))
            db.commit()
        endpoint = f"{path}/{first}/capture-day"
        body = {"day": "2025-08-15", "previous_day": None}
        self.assertEqual(self.http(path)[1][0]["id"], str(second))
        self.assertEqual(self.http(endpoint, "PATCH", body, token=None)[0], 401)
        self.assertEqual(self.http(endpoint.replace(str(self.participant_id), str(uuid4())), "PATCH", body)[0], 404)
        for invalid in ("1949-12-31", "2999-01-01"):
            self.assertEqual(self.http(endpoint, "PATCH", {**body, "day": invalid})[0], 422)
        for _ in range(2):
            self.assertEqual(self.http(endpoint, "PATCH", body)[0], 204)
        self.assertEqual(self.http(endpoint, "PATCH", {**body, "day": "2025-08-16"})[0], 409)
        row = self.http(f"/team/recordings/{first}")[1]
        self.assertEqual((row["captureDay"], row["heightAtCapture"], row["ageAtCapture"]), ("2025-08-15", 166.5, 75))
        self.assertEqual(row["live"]["record"]["captured_at"], "2025-08-15T00:00:00+09:00")
        self.assertEqual(self.http(path)[1][0]["id"], str(first))
        summary = self.http(f"/team/participant-summaries/{self.participant_id}")[1]
        self.assertEqual(summary["recentVisits"][0]["captured_at"], row["live"]["captured_at"])
        page = self.http(f"/team/participants/{self.participant_id}/recording-summaries")[1]
        self.assertEqual(page["items"][0]["id"], str(first))
        with Session(self.engine) as db:
            self.assertIsNone(db.get(GaitMeasurement, first).measured_at)
            self.assertEqual(db.scalar(select(GaitAnalysisResult).where(GaitAnalysisResult.measurement_id == first)).report_json, report)

    def test_shared_capture_survives_disconnect_and_replay(self):
        from unittest.mock import patch
        from app.ai_module import POSE_MODEL_SHA256
        from app.analysis_worker import process_one

        coords = self.body["keypoints_json"]
        coords.update(schema_version="team-local-coords/1", recording_id=self.key,
                      pose=dict(model="pose_landmarker_full", engine_version="1.2.3", delegate="CPU"),
                      marks=dict(stand_start=None, stand_end=None), frames=[dict(t=0, lm=None)])
        coords["capture"]["recorded_at"] = None
        packet = dict(requestId=self.key, coords=coords,
                      metadata=dict(recordedAtSource="unknown", rotationAppliedDegrees=0),
                      bridge_context=dict(profile="team-browser-full-cpu-video/1", source_sha256="unknown",
                                          model_sha256=POSE_MODEL_SHA256, input_object="ImageData",
                                          image_geometry="upright_unmirrored", clock_basis="video_pts",
                                          frame_audit_complete=True, fresh_tracker=True))
        body = dict(fileName="synthetic.mp4", packet=packet)
        path = f"/team/participants/{self.participant_id}/recordings"
        self.assertEqual(self.http(path, "POST", body, token=None)[0], 401)
        self.assertEqual(self.http(path, "POST", body)[0], 202)
        self.assertEqual(self.http(path, "POST", body)[0], 200)
        self.assertEqual(self.http(path, "POST", dict(body, fileName="other.mp4"))[0], 409)
        self.assertEqual(self.count(), 1)
        detail = f"/team/recordings/{self.key}"
        self.assertEqual(self.http(detail)[1]["serverState"], "pending")
        with Session(self.engine) as db, patch("app.analysis_worker.analyze_coords", side_effect=ValueError("test failure")):
            self.assertTrue(process_one(db))
        self.assertEqual(self.http(detail)[1]["serverState"], "failed")
        self.assertEqual(self.http(detail + "/retry", "POST")[0], 202)
        analysis = {"analysis": {"analysis_version": "test", "segments": []},
                    "record": {"captured_at": None}}
        with Session(self.engine) as db, patch("app.analysis_worker.analyze_coords", return_value=analysis) as run:
            self.assertTrue(process_one(db))
            self.assertFalse(process_one(db))
            run.assert_called_once_with(packet)
        row = self.http(detail)[1]
        self.assertEqual(row["serverState"], "ready")
        self.assertIsNone(row["ageAtCapture"])
        self.assertEqual(row["measuredAt"], "")
        self.assertEqual(row["live"]["id"], self.key)
        self.assertEqual(row["live"], self.http(path)[1][0]["live"])
        code, page, _ = self.http(f"/team/participants/{self.participant_id}/recording-summaries")
        self.assertEqual(code, 200)
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["preview"]["analysis"]["segments"], [])
        self.assertEqual(self.http(f"/measurements/{self.key}/result")[1]["analysis_result"]["report_json"], row["live"])
        self.assertEqual(self.http(path + "/" + self.key, "DELETE")[0], 204)
        self.assertEqual(self.http(detail)[0], 404)

    def test_team_summary_projection_and_legacy_compatibility(self):
        from app.models import GaitAnalysisResult
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ids = [uuid4() for _ in range(25)]
        graph = dict(schema_version="gait-cycle-degrees-1", left=[0]*51, right=[0]*51)
        report = {
            "captured_at": base.isoformat(), "source_sha256": "sample-source",
            "local_step_intervals_by_segment": {"s": [{"start_ms": 0, "stop_ms": 500, "landing_side": "left"}]},
            "analysis": {"segments": [{"segment_id": "s", "start_ms": 0, "stop_ms": 2000, "direction": "image_left",
                "knee": {"views": {"heavy_curve": list(range(2000))}},
                "gdi": {"models": {"normal100": {"status": "available", "value": {"left": 80, "right": 90}, "provenance": {"heavy": True}}}},
                "gait": {"steps": [], "quality_reasons": []}}],
                "current_report": {"segments": [{"segment_id": "s", "metrics": [
                    {"id": "rhythm", "evidence": {"tracking": {"transitions": [], "contact_radius_ms": 50}}},
                    {"id": "knee", "unneeded": True}]}]}}}
        with Session(self.engine) as db:
            for i, mid in enumerate(ids):
                db.add(GaitMeasurement(id=mid, participant_id=self.participant_id, age=76, created_by=self.user_id,
                    keypoints_json={"private_pose": [1,2,3]}, extractor_name="synthetic", extractor_version="1",
                    measured_at=base, analysis_started_at=base+timedelta(days=i)))
            db.flush()
            db.add(GaitAnalysisResult(measurement_id=ids[-1], hip_graph_json=graph,
                knee_graph_json={"schema_version": "unknown", "left": [0]}, ankle_graph_json=None,
                gdi_score=0, model_version="synthetic", analyzed_at=base))
            db.add(GaitAnalysisResult(measurement_id=ids[-2], report_json=report, model_version="synthetic", analyzed_at=base))
            db.commit()
        endpoint = f"/team/participants/{self.participant_id}/recording-summaries"
        code, page, _ = self.http(endpoint)
        self.assertEqual(code, 200)
        self.assertEqual((page["total"], page["offset"], page["limit"]), (25,0,20))
        self.assertEqual([m["id"] for m in page["items"]], [str(mid) for mid in reversed(ids[5:])])
        encoded = json.dumps(page)
        for name in ("private_pose", "heavy_curve", "provenance", "unneeded", "report_json", "hip_graph_json"):
            self.assertNotIn(name, encoded)
        preview = page["items"][1]["preview"]
        self.assertEqual(preview["analysis"]["segments"][0]["gdi"]["models"]["normal100"]["value"], {"left":80,"right":90})
        self.assertEqual(preview["local_step_intervals_by_segment"], report["local_step_intervals_by_segment"])
        self.assertEqual(len(self.http(endpoint+"?offset=20")[1]["items"]),5)
        self.assertEqual(self.http(endpoint+"?limit=21")[0],422)
        self.assertEqual(self.http(endpoint,token=None)[0],401)
        summary = self.http(f"/team/participant-summaries/{self.participant_id}")[1]
        self.assertEqual(summary["resultCount"], 25)
        self.assertEqual(len(summary["recentVisits"]), 1)
        visit = summary["recentVisits"][0]
        self.assertEqual(visit["id"], str(ids[-2]))
        self.assertEqual(visit["captured_at"], report["captured_at"])
        self.assertEqual(visit["source_sha256"], "sample-source")
        self.assertNotIn("heavy_curve", json.dumps(visit))
        self.assertNotIn("unneeded", json.dumps(visit))
        detail = self.http(f"/team/recordings/{ids[-1]}")[1]
        self.assertEqual(detail["dateBasis"], "analysis_start")
        self.assertTrue(detail["measuredAt"].startswith("2026-01-25"))
        self.assertEqual(detail["curves"]["hip"], {"left":[0]*51,"right":[0]*51})
        self.assertNotIn("knee", detail["curves"])
        self.assertIn("knee", detail["missing"])
        self.assertIn("ankle", detail["missing"])

    def test_summary_endpoints_are_bounded_and_exclude_heavy_json(self):
        endpoint = f"/participants/{self.participant_id}"
        code, empty, _ = self.http(endpoint + "/summary")
        self.assertEqual(code, 200)
        self.assertEqual(empty["measurement_count"], 0)
        self.assertIsNone(empty["latest_measured_at"])
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ids = [uuid4() for _ in range(25)]
        with Session(self.engine) as db:
            for i, mid in enumerate(ids):
                db.add(GaitMeasurement(
                    id=mid, participant_id=self.participant_id, age=76, created_by=self.user_id,
                    keypoints_json={"large_private_pose": [1, 2, 3]}, extractor_name="synthetic", extractor_version="1",
                    measured_at=base + timedelta(days=i) if i < 24 else None,
                    analysis_started_at=base + timedelta(days=i) if i == 24 else None,
                ))
            db.commit()
        with self.engine.begin() as conn:
            conn.execute(text("""INSERT INTO gait_analysis_results
                (id,measurement_id,gdi_score,cadence_spm,stride_length_cm,hip_graph_json,model_version,analyzed_at)
                VALUES (:id,:mid,0,NULL,0,'{"large_graph":[1,2,3]}'::jsonb,'synthetic',now())"""),
                {"id": uuid4(), "mid": ids[-1]})
        code, summaries, _ = self.http("/participants/summaries")
        self.assertEqual(code, 200)
        summary = next(row for row in summaries if row["id"] == str(self.participant_id))
        self.assertEqual(summary["measurement_count"], 25)
        self.assertEqual(summary["latest_measured_at"], "2026-01-25T00:00:00Z")
        self.assertEqual(self.http(endpoint + "/summary")[1], summary)
        code, first, _ = self.http(endpoint + "/measurement-summaries")
        self.assertEqual(code, 200)
        self.assertEqual((first["total"], first["offset"], first["limit"]), (25, 0, 20))
        self.assertEqual([row["id"] for row in first["items"]], [str(mid) for mid in reversed(ids[5:])])
        self.assertEqual(first["items"][0]["gdi_score"], 0)
        self.assertIsNone(first["items"][0]["cadence_spm"])
        self.assertTrue(first["items"][0]["has_analysis_result"])
        self.assertFalse(first["items"][1]["has_analysis_result"])
        self.assertIsNone(first["items"][1]["gdi_score"])
        code, second, _ = self.http(endpoint + "/measurement-summaries?offset=20")
        self.assertEqual(code, 200)
        self.assertEqual([row["id"] for row in second["items"]], [str(mid) for mid in reversed(ids[:5])])
        for row in first["items"] + second["items"]:
            for private in ["keypoints_json", "processing_metadata", "hip_graph_json", "knee_graph_json", "ankle_graph_json"]:
                self.assertNotIn(private, row)
        self.assertEqual(self.http(endpoint + "/measurement-summaries?offset=100")[1]["items"], [])
        for route in ["/participants/summaries", endpoint + "/summary", endpoint + "/measurement-summaries"]:
            self.assertEqual(self.http(route, token=None)[0], 401)
        self.assertEqual(self.http(f"/participants/{uuid4()}/summary")[0], 404)
        self.assertEqual(self.http(f"/participants/{uuid4()}/measurement-summaries")[0], 404)
        self.assertEqual(self.http(endpoint + "/measurement-summaries?limit=101")[0], 422)

    def test_migration_preserves_existing_data(self):
        with Session(self.engine) as db:
            old = db.get(GaitMeasurement, self.legacy_measurement)
            self.assertEqual(old.age, 75)
            self.assertIsNone(old.processing_metadata)
            self.assertIsNotNone(db.get(User, self.legacy_user))
            self.assertIsNotNone(db.get(Participant, self.legacy_participant))

    def test_success_login_and_metadata_storage(self):
        code, login, _ = self.http("/auth/login", "POST", urlencode({"username": self.login_id, "password": self.password}).encode(),
                                  token=None, headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(code, 200)
        code, result, _ = self.http(self.path, "POST", self.body, self.key, login["access_token"])
        self.assertEqual(code, 201)
        with Session(self.engine) as db:
            row = db.get(GaitMeasurement, UUID(result["id"]))
            self.assertEqual(row.processing_metadata, self.body["processing_metadata"])
            self.assertEqual(row.keypoints_json, self.body["keypoints_json"])
        code, rows, _ = self.http(self.path)
        self.assertEqual(code, 200)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("keypoints_json", rows[0])
        self.assertNotIn("processing_metadata", rows[0])
        self.assertFalse(rows[0]["has_analysis_result"])
        code, result, _ = self.http(f'/measurements/{result["id"]}/result')
        self.assertEqual(code, 200)
        self.assertEqual(result["status"], "not_available")

    def test_replay_same_body_and_fresh_token_returns_same_response(self):
        one = self.http(self.path, "POST", self.body, self.key)
        with Session(self.engine) as db:
            db.get(Participant, self.participant_id).birth_date = date(2000, 1, 1)
            db.commit()
        two = self.http(self.path, "POST", self.body, self.key, create_access_token(self.user_id))
        self.assertEqual((one[0], two[0]), (201, 200))
        self.assertEqual(one[1], two[1])
        self.assertEqual(self.count(), 1)

    def test_same_key_different_bytes_conflicts(self):
        self.assertEqual(self.http(self.path, "POST", self.body, self.key)[0], 201)
        spaced = json.dumps(self.body, indent=2).encode()
        code, error, _ = self.http(self.path, "POST", spaced, self.key)
        self.assertEqual(code, 409)
        self.assertEqual(error["detail"]["code"], "IDEMPOTENCY_CONFLICT")
        self.assertEqual(self.count(), 1)

    def test_same_key_is_scoped_to_user(self):
        self.assertEqual(self.http(self.path, "POST", self.body, self.key)[0], 201)
        self.assertEqual(self.http(self.path, "POST", self.body, self.key, self.other_token)[0], 201)
        self.assertEqual(self.count(), 2)

    def test_key_cannot_be_reused_for_a_different_participant(self):
        self.assertEqual(self.http(self.path, "POST", self.body, self.key)[0], 201)
        other = uuid4()
        with Session(self.engine) as db:
            db.add(Participant(id=other, created_by=self.user_id, name="Other synthetic participant", birth_date=date(1950, 1, 1)))
            db.commit()
        self.body["keypoints_json"]["subject"]["id"] = str(other)
        code, error, _ = self.http(f"/participants/{other}/measurements", "POST", self.body, self.key)
        self.assertEqual((code, error["detail"]["code"]), (409, "IDEMPOTENCY_CONFLICT"))
        with Session(self.engine) as db:
            self.assertEqual(db.scalar(select(func.count()).select_from(GaitMeasurement).where(GaitMeasurement.participant_id == other)), 0)

    def test_korean_date_boundary_and_before_birth(self):
        self.body["measured_at"] = "2025-12-31T15:00:00Z"
        self.body["keypoints_json"]["capture"]["recorded_at"] = self.body["measured_at"]
        code, row, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual((code, row["age"]), (201, 76))
        self.body["measured_at"] = "1949-01-01T00:00:00Z"
        self.body["keypoints_json"]["capture"]["recorded_at"] = self.body["measured_at"]
        self.assertEqual(self.http(self.path, "POST", self.body, str(uuid4()))[0], 422)

    def test_nonfinite_validation_error_does_not_echo_coordinates(self):
        self.body["keypoints_json"]["frames"][0]["lm"][0][0] = float("nan")
        code, error, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual(code, 422)
        self.assertTrue(all("input" not in item for item in error["detail"]))
        self.assertEqual(self.count(), 0)

    def test_concurrent_same_key_and_body_creates_one_record(self):
        barrier = threading.Barrier(2)
        def submit():
            barrier.wait(timeout=5)
            return self.http(self.path, "POST", self.body, self.key)
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = [future.result() for future in [pool.submit(submit), pool.submit(submit)]]
        self.assertEqual(sorted([a[0], b[0]]), [200, 201])
        self.assertEqual(a[1]["id"], b[1]["id"])
        self.assertEqual(self.count(), 1)

    def test_lock_timeout_rolls_back_losing_measurement_and_can_retry(self):
        raw = json.dumps(self.body, separators=(",", ":")).encode()
        with Session(self.engine) as holder:
            # An uncommitted real uniqueness conflict, not a mocked DB exception.
            holder.add(MeasurementRequest(user_id=self.user_id, request_key=UUID(self.key),
                        participant_id=self.participant_id, body_sha256=hashlib.sha256(raw).hexdigest(),
                        measurement_id=self.legacy_measurement))
            holder.flush()
            code, error, headers = self.http(self.path, "POST", raw, self.key,
                                             headers={"Origin": "http://localhost:5173"})
            self.assertEqual(code, 409)
            self.assertEqual(error["detail"]["code"], "REQUEST_IN_PROGRESS")
            self.assertEqual(headers["Retry-After"], "2")
            self.assertIn("Retry-After", headers["Access-Control-Expose-Headers"])
            self.assertEqual(self.count(), 0)
            holder.rollback()
        self.assertEqual(self.http(self.path, "POST", raw, self.key)[0], 201)

    def test_invalid_request_does_not_reserve_key(self):
        bad = copy.deepcopy(self.body)
        bad["keypoints_json"]["frames"][1]["t"] = 0
        self.assertEqual(self.http(self.path, "POST", bad, self.key)[0], 422)
        self.assertEqual(self.count(), 0)
        self.assertEqual(self.http(self.path, "POST", self.body, self.key)[0], 201)

    def test_auth_pairing_and_subject_validation(self):
        self.assertEqual(self.http(self.path, "POST", self.body, self.key, token=None)[0], 401)
        self.assertEqual(self.http(self.path, "POST", self.body, self.key, token="invalid")[0], 401)
        self.assertEqual(self.http(self.path, "POST", self.body)[0], 422)
        legacy = copy.deepcopy(self.body)
        del legacy["processing_metadata"]
        self.assertEqual(self.http(self.path, "POST", legacy, self.key)[0], 422)
        self.assertEqual(self.http(self.path, "POST", self.body, "not-uuid")[0], 422)
        self.body["keypoints_json"]["subject"]["id"] = str(uuid4())
        code, error, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual(code, 422)
        self.assertEqual(error["detail"]["code"], "PARTICIPANT_MISMATCH")
        self.assertEqual(self.count(), 0)

    def test_legacy_request_and_missing_birth_date(self):
        with Session(self.engine) as db:
            db.get(Participant, self.participant_id).birth_date = None
            db.commit()
        code, error, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual((code, error["detail"]["code"]), (409, "PARTICIPANT_BIRTH_DATE_REQUIRED"))
        with Session(self.engine) as db:
            db.get(Participant, self.participant_id).birth_date = date(1950, 1, 1)
            db.commit()
        del self.body["processing_metadata"]
        self.body["keypoints_json"] = {"legacy": True}
        self.assertEqual(self.http(self.path, "POST", self.body)[0], 201)

    def test_cors_allows_request_key_and_rejects_untrusted_origin(self):
        headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                   "Access-Control-Request-Headers": "authorization,content-type,idempotency-key"}
        code, _, response_headers = self.http(self.path, "OPTIONS", token=None, headers=headers)
        self.assertEqual(code, 200)
        self.assertEqual(response_headers["Access-Control-Allow-Origin"], headers["Origin"])
        headers["Origin"] = "https://untrusted.example"
        self.assertEqual(self.http(self.path, "OPTIONS", token=None, headers=headers)[0], 400)


if __name__ == "__main__":
    unittest.main()
