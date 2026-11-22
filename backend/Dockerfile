FROM python:3.12-slim AS base
WORKDIR /service/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS api
RUN apt-get update && apt-get install -y --no-install-recommends libgles2 libegl1 libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/video-pose \
    && /opt/video-pose/bin/pip install --no-cache-dir mediapipe==1.2.3 av==19.0.2
COPY backend/ ./
COPY ai-module/ /service/ai-module/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS analysis
COPY ai-module/requirements.txt /tmp/analysis-requirements.txt
RUN pip install --no-cache-dir -r /tmp/analysis-requirements.txt && pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY backend/ ./
COPY ai-module/ /service/ai-module/
COPY research/selected/frozen-gdi/ /service/research/selected/frozen-gdi/
COPY research/paper-related/normal100-models/models.json /service/research/paper-related/normal100-models/models.json
CMD ["python", "-m", "app.analysis_worker"]
