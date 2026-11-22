"""Run from a Docker host with openssl. Uses a synthetic clip and disposable login."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import secrets
import socket
import ssl
import subprocess
import tempfile
import time


def docker(*args, stdin=None):
    return subprocess.run(["docker", *args], input=stdin, stdout=subprocess.PIPE, check=True).stdout


def eventually(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError("Expected descriptor count not observed")


def verify(image, models, root):
    name = "video-privacy-" + secrets.token_hex(5)
    tls = name + "-tls"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost",
                    "-keyout", str(root/"key"), "-out", str(root/"cert")], capture_output=True, check=True)
    snippet = Path(__file__).resolve().parents[1] / "deploy/video-pose-tls.conf"
    (root/"test.conf").write_text("server { listen 443 ssl; ssl_certificate /test/cert; "
                                  "ssl_certificate_key /test/key; " + snippet.read_text() + "}")
    try:
        docker("run", "-d", "--name", name, "-p", "127.0.0.1::443",
               "-e", "DATABASE_URL=sqlite:////dev/shm/users", "-e", "JWT_SECRET_KEY="+secrets.token_urlsafe(32),
               "-v", f"{models}:/models:ro", image)
        docker("run", "-d", "--name", tls, "--network", "container:"+name, "-v", f"{root}:/test:ro",
               "-v", f"{root/'test.conf'}:/etc/nginx/conf.d/default.conf:ro", "nginx:stable-alpine")
        port = int(docker("port", name, "443").decode().strip().rsplit(":", 1)[1])
        ctx = ssl.create_default_context(cafile=str(root/"cert"))
        path = "/api/team/video-pose"
        token = docker("exec", "-i", name, "python", "-", stdin=b"""
from app.db import engine, SessionLocal
from app.models import User
from app.security import create_access_token
User.__table__.create(engine)
with SessionLocal() as db:
    user = User(login_id='synthetic-test', password_hash='unused')
    db.add(user); db.commit(); print(create_access_token(user.id))
""").decode().strip()
        auth = {"Authorization": "Bearer "+token, "Content-Type": "application/octet-stream"}
        def request(method, data=None, headers=auth):
            c = http.client.HTTPSConnection("localhost", port, context=ctx, timeout=90)
            try:
                c.request(method, path, body=data, headers=headers)
                r = c.getresponse()
                return r.status, r.read()
            finally:
                c.close()

        assert request("GET", headers={})[0] == 401
        assert request("GET")[0] == 200
        assert request("POST", b"invalid", headers={})[0] == 401
        clip = docker("exec", "-i", name, "/opt/video-pose/bin/python", "-", stdin=b"""
import av, io, numpy as np, sys
b = io.BytesIO()
with av.open(b, mode='w', format='mp4') as out:
    s = out.add_stream('mpeg4', rate=30); s.width=160; s.height=120; s.pix_fmt='yuv420p'
    for i in range(90):
        f = av.VideoFrame.from_ndarray(np.full((120,160,3),i,dtype=np.uint8), format='rgb24')
        for packet in s.encode(f): out.mux(packet)
    for packet in s.encode(): out.mux(packet)
sys.stdout.buffer.write(b.getvalue())
""")
        def handles():
            return int(docker("exec", name, "sh", "-c",
                              "ls -l /proc/[0-9]*/fd/ 2>/dev/null | grep -c '/memfd:happywalk-video' || :").strip())

        for data, expected in [(clip, 200), (b"not-a-video", 422)]:
            with ThreadPoolExecutor(1) as pool:
                job = pool.submit(request, "POST", data)
                eventually(lambda: handles() > 0)
                status, body = job.result()
            assert status == expected, (status, body[:120])
            assert handles() == 0
            if status == 200:
                result = json.loads(body)
                assert len(result["frames"]) == 90 and result["source_sha256"] == hashlib.sha256(data).hexdigest()
            print(f"HTTP {status}: observed live video, zero handles at response", flush=True)

        for stage in ("upload", "analysis"):
            c = http.client.HTTPSConnection("localhost", port, context=ctx)
            c.connect()
            c.putrequest("POST", path)
            for k, v in auth.items(): c.putheader(k, v)
            c.putheader("Content-Length", str(len(clip)))
            c.endheaders(clip[:200] if stage == "upload" else clip)
            eventually(lambda: handles() > 0)
            if stage == "analysis":
                eventually(lambda: tracker_running(name))
            c.sock.shutdown(socket.SHUT_RDWR)
            c.close()
            eventually(lambda: handles() == 0, timeout=3)
            assert not tracker_running(name)
            print(f"Cancel during {stage}: zero handles within 3 s", flush=True)
        assert not docker("exec", tls, "find", "/var/cache/nginx/client_temp", "-type", "f").strip()
        print("Certificate verified, no nginx body files, anonymous requests rejected")
    finally:
        for container in (tls, name):
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)


def tracker_running(name):
    return b"video_pose_process.py" in docker("top", name, "-eo", "pid,args")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("models", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        verify(args.image, args.models.resolve(), Path(directory).resolve())
