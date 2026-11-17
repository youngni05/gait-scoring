"""Repeatable CI checks. Uses DATABASE_URL; performs no application data writes."""

import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4


def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL must point to a test database")
    # Per-run test key, not a repository or deployment secret.
    os.environ["JWT_SECRET_KEY"] = secrets.token_urlsafe(32)
    os.environ["CORS_ALLOW_ORIGINS"] = "http://localhost:5173"
    from app.security import create_access_token, hash_password, verify_password
    from app.config import JWT_SECRET_KEY
    import jwt

    password = secrets.token_urlsafe(16)
    hashed = hash_password(password)
    assert verify_password(password, hashed)
    assert not verify_password(password + "wrong", hashed)
    uid = uuid4()
    assert jwt.decode(create_access_token(uid), JWT_SECRET_KEY, algorithms=["HS256"])["sub"] == str(uid)

    for invalid_key in ("", " " * 40, "replace_with_a_random_secret", "x" * 31):
        result = subprocess.run(
            [sys.executable, "-c", "import app.main"],
            env={**os.environ, "JWT_SECRET_KEY": invalid_key},
            capture_output=True, text=True, timeout=20,
        )
        assert result.returncode != 0 and "JWT_SECRET_KEY must" in result.stderr

    port = 8765
    base_url = f"http://127.0.0.1:{port}"

    def request(path, method="GET", headers=None, data=None):
        req = Request(base_url + path, method=method, headers=headers or {}, data=data)
        try:
            response = urlopen(req, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, response.read()

    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError("API process exited before readiness")
                try:
                    status, _, body = request("/health")
                    if status == 200:
                        break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("API readiness timed out")
                time.sleep(0.2)
            assert json.loads(body) == {"status": "ok"}
            status, _, body = request("/db-health")
            assert status == 200 and json.loads(body) == {"database": "connected"}
            status, _, body = request("/openapi.json")
            assert status == 200
            paths = json.loads(body)["paths"]
            for path, methods in {
                "/auth/login": {"post"}, "/auth/me": {"get"},
                "/participants": {"get", "post"},
                "/participants/{participant_id}": {"get", "patch"},
                "/participants/{participant_id}/measurements": {"get", "post"},
                "/measurements/{measurement_id}/result": {"get"},
            }.items():
                assert methods <= paths[path].keys(), path
            assert request("/participants")[0] == 401
            assert request("/auth/me", headers={"Authorization": "Bearer invalid"})[0] == 401
            status, _, _ = request(
                "/auth/login", "POST", {"Content-Type": "application/x-www-form-urlencoded"},
                urlencode({"username": str(uuid4()), "password": password}).encode(),
            )
            assert status == 401
            status, headers, _ = request("/participants", "OPTIONS", {
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Headers": "authorization,content-type",
            })
            assert status == 200 and headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
            status, headers, _ = request("/participants", "OPTIONS", {
                "Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST",
            })
            assert status == 400 and headers.get("Access-Control-Allow-Origin") is None
            print("PASS: JWT settings, password hashing, API/DB health, OpenAPI routes, authentication and CORS")
        except Exception:
            log.seek(0)
            print(log.read().decode(errors="replace"), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
