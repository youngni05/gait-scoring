import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, auth, participants, measurements

app = FastAPI(
    title="Happy Aging API",
    version="0.1.0",
    description="해피에이징 보행 영상 수집 및 분석 서비스 API",
)

cors_origins = list(dict.fromkeys(
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
))
if "*" in cors_origins:
    raise ValueError("CORS_ALLOW_ORIGINS must list explicit frontend origins, not '*'")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(participants.router)
app.include_router(measurements.router)
