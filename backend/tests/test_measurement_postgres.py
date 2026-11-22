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
import jwt
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

    def test_expired_token_rejects_reads_and_writes_without_side_effects(self):
        code, measurement, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual(code, 201)
        expired = jwt.encode(
            {"sub": str(self.user_id), "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
            os.environ["JWT_SECRET_KEY"], algorithm="HS256",
        )
        participant_path = f"/participants/{self.participant_id}"
        new_key = str(uuid4())
        requests = [
            ("GET", "/auth/me", None, None),
            ("GET", "/participants", None, None),
            ("GET", participant_path, None, None),
            ("GET", self.path, None, None),
            ("GET", f'/measurements/{measurement["id"]}/result', None, None),
            ("POST", "/participants", {"name": "Unauthorized fixture", "gender": "M",
                                       "birth_date": "1950-01-01", "height_cm": "170.00"}, None),
            ("PATCH", participant_path, {"name": "Unauthorized edit"}, None),
            ("POST", self.path, self.body, new_key),
        ]
        with Session(self.engine) as db:
            participants_before = db.scalar(select(func.count()).select_from(Participant))
        for method, path, body, key in requests:
            with self.subTest(method=method, path=path):
                code, error, headers = self.http(path, method, body, key, token=expired)
                self.assertEqual(code, 401)
                self.assertEqual(error["detail"], "Invalid or expired token")
                self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        with Session(self.engine) as db:
            self.assertEqual(db.scalar(select(func.count()).select_from(Participant)), participants_before)
            self.assertEqual(db.get(Participant, self.participant_id).name, "Synthetic participant")
            self.assertIsNone(db.get(MeasurementRequest, (self.user_id, UUID(new_key))))
        self.assertEqual(self.count(), 1)
        # A real login must restore access; the rejected request did not consume its key.
        code, login, _ = self.http("/auth/login", "POST",
            urlencode({"username": self.login_id, "password": self.password}).encode(),
            token=None, headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(code, 200)
        self.assertEqual(self.http(self.path, "POST", self.body, new_key, login["access_token"])[0], 201)
        self.assertEqual(self.count(), 2)

    def test_participant_patch_by_other_worker_preserves_omitted_fields(self):
        path = f"/participants/{self.participant_id}"
        code, before, _ = self.http(path)
        self.assertEqual(code, 200)
        code, updated, _ = self.http(path, "PATCH", {"name": "  Updated fixture  ", "height_cm": "180.25"},
                                      token=self.other_token)
        self.assertEqual(code, 200)
        expected = {**before, "name": "Updated fixture", "height_cm": "180.25"}
        self.assertEqual(updated, expected)
        self.assertEqual(updated["created_by"], str(self.user_id))
        self.assertEqual(self.http(path)[1], expected)
        with Session(self.engine) as db:
            row = db.get(Participant, self.participant_id)
            self.assertEqual(row.name, "Updated fixture")
            self.assertEqual(str(row.height_cm), "180.25")
            self.assertEqual(row.birth_date, date(1950, 1, 1))

    def test_participant_patch_invalid_input_is_atomic(self):
        path = f"/participants/{self.participant_id}"
        before = self.http(path)[1]
        future = (datetime.now(timezone(timedelta(hours=9))).date() + timedelta(days=1)).isoformat()
        invalid = [{}, {"name": " "}, {"name": "x" * 101}, {"gender": "X"},
                   {"birth_date": future}, {"birth_date": "1950-02-30"},
                   {"height_cm": "0"}, {"height_cm": "170.123"},
                   {"created_by": str(self.other_user_id)}, {"id": str(uuid4())}]
        invalid.extend({field: None} for field in ("name", "gender", "birth_date", "height_cm"))
        # Include a valid change with an invalid one to detect partial writes.
        invalid.append({"name": "Must not persist", "height_cm": "-1"})
        for payload in invalid:
            with self.subTest(payload=payload):
                code, error, _ = self.http(path, "PATCH", payload)
                self.assertEqual(code, 422)
                self.assertIsInstance(error["detail"], list)
                self.assertEqual(self.http(path)[1], before)
        code, error, _ = self.http(f"/participants/{uuid4()}", "PATCH", {"name": "Missing fixture"})
        self.assertEqual((code, error["detail"]), (404, "Participant not found"))

    def test_birth_date_patch_preserves_history_and_changes_new_measurement_age(self):
        # The same capture instant is Jan 1 in Korea, before the corrected Jan 2 birthday.
        self.body["measured_at"] = "2025-12-31T15:00:00Z"
        self.body["keypoints_json"]["capture"]["recorded_at"] = self.body["measured_at"]
        code, original, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual((code, original["age"]), (201, 76))
        code, _, _ = self.http(f"/participants/{self.participant_id}", "PATCH",
            {"birth_date": "1960-01-02", "height_cm": "180.25"}, token=self.other_token)
        self.assertEqual(code, 200)
        code, replay, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual(code, 200)
        self.assertEqual(replay, original)
        code, new, _ = self.http(self.path, "POST", self.body, str(uuid4()))
        self.assertEqual((code, new["age"]), (201, 65))
        code, rows, _ = self.http(self.path, token=self.other_token)
        self.assertEqual(code, 200)
        self.assertEqual({row["id"]: row["age"] for row in rows}, {original["id"]: 76, new["id"]: 65})
        code, result, _ = self.http(f'/measurements/{original["id"]}/result', token=self.other_token)
        self.assertEqual(code, 200)
        self.assertEqual(result["measurement"]["age"], 76)
        self.assertEqual(result["participant"]["birth_date"], "1960-01-02")
        self.assertEqual(result["participant"]["height_cm"], "180.25")
        with Session(self.engine) as db:
            self.assertEqual(db.get(GaitMeasurement, UUID(original["id"])).age, 76)
            self.assertEqual(db.get(GaitMeasurement, UUID(new["id"])).age, 65)

    def test_partial_analysis_result_preserves_null_zero_and_graph(self):
        code, measurement, _ = self.http(self.path, "POST", self.body, self.key)
        self.assertEqual(code, 201)
        path = f'/measurements/{measurement["id"]}/result'
        code, before, _ = self.http(path)
        self.assertEqual(code, 200)
        self.assertEqual(before["status"], "not_available")
        self.assertIsNone(before["analysis_result"])
        # Synthetic storage fixture only: no AI endpoint or finalized graph contract.
        graph = {"fixture": "synthetic", "samples": [0, None, 12.5]}
        result_id = uuid4()
        analyzed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with Session(self.engine) as db:
            db.add(GaitAnalysisResult(id=result_id, measurement_id=UUID(measurement["id"]),
                gdi_score=0.0, stride_length_cm=None, cadence_spm=98.5,
                hip_graph_json=None, knee_graph_json=graph, ankle_graph_json=None,
                model_version="synthetic-regression-v1", analyzed_at=analyzed_at))
            db.commit()
        code, result, _ = self.http(path, token=self.other_token)
        self.assertEqual(code, 200)
        self.assertEqual(result["status"], "available")
        analysis = result["analysis_result"]
        self.assertEqual(set(analysis), {"id", "measurement_id", "gdi_score", "stride_length_cm",
            "cadence_spm", "hip_graph_json", "knee_graph_json", "ankle_graph_json", "model_version", "analyzed_at"})
        self.assertEqual(analysis["id"], str(result_id))
        self.assertEqual(analysis["measurement_id"], measurement["id"])
        self.assertEqual(analysis["gdi_score"], 0.0)
        self.assertEqual(analysis["cadence_spm"], 98.5)
        for field in ("stride_length_cm", "hip_graph_json", "ankle_graph_json"):
            self.assertIsNone(analysis[field])
        self.assertEqual(analysis["knee_graph_json"], graph)
        self.assertEqual(analysis["model_version"], "synthetic-regression-v1")
        self.assertEqual(datetime.fromisoformat(analysis["analyzed_at"].replace("Z", "+00:00")), analyzed_at)
        self.assertEqual(result["measurement"], measurement)
        self.assertNotIn("keypoints_json", result["measurement"])
        self.assertNotIn("processing_metadata", result["measurement"])
        code, rows, _ = self.http(self.path, token=self.other_token)
        self.assertEqual(code, 200)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["has_analysis_result"])
        self.assertEqual(rows[0]["id"], measurement["id"])


if __name__ == "__main__":
    unittest.main()
