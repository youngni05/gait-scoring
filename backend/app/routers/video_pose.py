"""Authenticated streaming upload. Never give the original a filesystem path."""
import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from starlette.requests import ClientDisconnect

from app.ai_module import POSE_MODEL_SHA256
from app.deps import get_current_user
from app.routers.health import verified_pose_model

router = APIRouter(prefix="/team", dependencies=[Depends(get_current_user)])
MAX_BYTES = 64 * 1024 * 1024
EXTRACTOR = Path(__file__).resolve().parents[1] / "video_pose_process.py"
PYTHON = "/opt/video-pose/bin/python"
slot = asyncio.Lock()


def ready(request: Request):
    if request.headers.get("x-happywalk-video-transport") != "tls-stream-v1":
        raise HTTPException(403, "Streaming HTTPS ingress required")
    try:
        return verified_pose_model()
    except (ValueError, OSError):
        raise HTTPException(503, "Pose model unavailable") from None


@router.get("/video-pose")
def capabilities(request: Request):
    ready(request)
    return {"transport": "tls-stream-v1", "max_bytes": MAX_BYTES, "model_sha256": POSE_MODEL_SHA256}


async def extract(request, video, model):
    # ffmpeg and the tracker run without database credentials or application tokens.
    proc = await asyncio.create_subprocess_exec(
        PYTHON, "-B", str(EXTRACTOR), str(model), str(os.getpid()), stdin=video,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        env={"HOME": "/nonexistent", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
             "MPLCONFIGDIR": "/dev/shm/happywalk-mpl"},
    )
    output = asyncio.create_task(proc.communicate())
    try:
        while not output.done():
            if await request.is_disconnected():
                raise ClientDisconnect()
            await asyncio.wait({output}, timeout=0.1)
        data, _ = output.result()
        if proc.returncode or not data:
            raise HTTPException(422, "Video extraction failed")
        return Response(data, media_type="application/json", headers={"Cache-Control": "no-store"})
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await output


@router.post("/video-pose")
async def video_pose(request: Request):
    model = ready(request)
    if request.headers.get("content-type") != "application/octet-stream":
        raise HTTPException(415, "Expected raw video")
    if slot.locked():
        raise HTTPException(429, "CPU busy", headers={"Retry-After": "5"})
    async with slot:
        try:
            async with asyncio.timeout(165):
                with os.fdopen(os.memfd_create("happywalk-video"), "w+b") as video:
                    async for chunk in request.stream():
                        if video.tell() + len(chunk) > MAX_BYTES:
                            raise HTTPException(413, "Video too large")
                        video.write(chunk)
                    del chunk
                    if not video.tell():
                        raise HTTPException(422, "Empty video")
                    video.seek(0)
                    return await extract(request, video, model)
        except TimeoutError:
            raise HTTPException(408, "Video processing timed out") from None
        except ClientDisconnect:
            return Response(status_code=499)
