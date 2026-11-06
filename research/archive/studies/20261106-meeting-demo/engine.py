"""Local inference using the existing MediaPipe adapter and frozen GDI features."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MP_HOME = ROOT / 'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot'
ASSETS = Path('D:/gait-research-data/D_P/note20-score-20261102/assets')
sys.path[:0] = [str(MP_HOME / 'deps'), str(ASSETS.parent / 'deps')]
import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as ort
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

pre = module('meeting_frozen_preprocess', ROOT / 'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle/preprocess.py')
adapter = module('meeting_existing_adapter', ROOT / 'gdi-research-harness/studies/20261101-self-video-mediapipe/common.py')
MANIFEST = json.loads((ASSETS / 'manifest.json').read_text(encoding='utf-8'))
TASK = MP_HOME / 'pose_landmarker_full.task'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_assets():
    if MANIFEST['predictor'] != 'frozen54-fusion' or len(MANIFEST['members']) != 54:
        raise ValueError('기존 54개 모델 구성이 아니다.')
    for m in MANIFEST['members']:
        if sha(ASSETS / m['path']) != m['sha256']:
            raise ValueError('모델 파일 지문이 다르다: ' + m['id'])
    expected = json.loads((MP_HOME / 'pose_landmarker_full.task.http.json').read_text(encoding='utf-8-sig'))['sha256']
    if sha(TASK) != expected:
        raise ValueError('MediaPipe 모델 파일 지문이 다르다.')

def extract(path, progress=lambda **kw: None):
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('영상을 열 수 없다. MP4(H.264) 파일로 다시 선택한다.')
        if not 29 <= fps <= 31:
            raise ValueError('이번 시연은 기존 입력 조건인 30fps 영상을 지원한다. 30fps로 촬영한 영상을 선택한다.')
        if count < 124 or count > 1499:
            raise ValueError('영상은 124~1499프레임(30fps 기준 약 4.2~49.9초)이어야 한다.')
        width, height = int(cap.get(3)), int(cap.get(4))
        landmarks = np.full((count, 33, 5), np.nan)
        times = []
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(TASK), delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
            min_tracking_confidence=.5, output_segmentation_masks=False)
        cv2.setNumThreads(2)
        previous = -1
        with vision.PoseLandmarker.create_from_options(options) as detector:
            for i in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('영상 프레임을 끝까지 읽지 못했다. 다른 파일을 선택한다.')
                source_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if not np.isfinite(source_ms) or (times and source_ms < times[-1] * 1000):
                    raise ValueError('영상의 시간 순서를 확인할 수 없다.')
                times.append(source_ms / 1000)
                timestamp = max(int(round(source_ms)), previous + 1)
                previous = timestamp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                if result.pose_landmarks:
                    landmarks[i] = [[p.x, p.y, p.z, p.visibility, p.presence] for p in result.pose_landmarks[0]]
                if i % 15 == 0:
                    progress(stage='pose', done=i + 1, total=count, message='MediaPipe로 관절점을 찾는 중')
            if cap.read()[0]:
                raise ValueError('영상의 프레임 수가 기록과 다르다.')
        raw = adapter.adapt(landmarks, width, height)
        try:
            features = pre.extract_windows(raw)
        except ValueError as e:
            raise ValueError('측정 불가: 충분한 관절점과 124프레임 연속 구간이 필요하다. 전신이 보이는 영상을 선택한다.') from e
        joints = [23, 25, 27, 31, 24, 26, 28, 32]
        view = landmarks[:, joints][:, :, [0, 1, 3]]
        # JSON null represents a missing point; never turn a missing pose into 0 GDI.
        overlay = [[[None if not np.isfinite(v) else round(float(v), 5) for v in pt] for pt in frame] for frame in view]
        low = np.any(~np.isfinite(view[:, :, 2]) | (view[:, :, 2] < .5), axis=1)
        review = dict(width=width, height=height, fps=fps, frames=count, seconds=count/fps,
                      times=times, overlay=overlay, windows=len(features['x']),
                      starts=features['window_start'].tolist(),
                      missing=features['window_missing_fraction'].tolist(),
                      lowVisibilityFrames=int(low.sum()), elapsed=time.perf_counter()-started,
                      extractor='MediaPipe Full', input='2D image coordinates',
                      note='MediaPipe 발끝은 학습 자료의 엄지발가락과 근사 대응한다.')
        return features['x'], review
    finally:
        cap.release()

def predict(x, progress=lambda **kw: None):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (2, 124, 12) or not 1 <= len(x) <= 13 or not np.isfinite(x).all():
        raise ValueError('고정 모델 입력 형식에 맞지 않는다.')
    started = time.perf_counter()
    values = {}
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    for index, m in enumerate(MANIFEST['members']):
        path = ASSETS / m['path']
        if sha(path) != m['sha256']:
            raise ValueError('모델 파일이 변경되었다.')
        session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        raw = session.run(['raw_scores'], {'features': x})[0]
        if raw.shape != (len(x), 2) or not np.isfinite(raw).all():
            raise ValueError('모델 출력이 올바르지 않다.')
        values[m['id']] = raw.astype(np.float64).mean(0) * m['calibration'][0] + m['calibration'][1]
        del session
        progress(stage='score', done=index+1, total=109, message='기존 학습 모델 계산 중')
    folds = []
    for selection in MANIFEST['selections']:
        means = {family: np.mean([values[m['id']] for m in MANIFEST['members']
                 if m['family'] == family and m['fold'] == selection['outer']], axis=0)
                 for family in ('CNN', 'STT', 'bilateral')}
        base = selection['cnn_weight'] * means['CNN'] + (1-selection['cnn_weight']) * means['STT']
        delta = (1-selection['fraction'])*(base[0]-base[1]) + selection['fraction']*(means['bilateral'][0]-means['bilateral'][1])
        folds.append([base.mean()+delta/2, base.mean()-delta/2])
    result = np.mean(folds, axis=0)
    if not np.isfinite(result).all():
        raise ValueError('점수가 유효하지 않다.')
    from comparison import compare
    comparison = compare(x, values, progress)
    return dict(left=float(result[0]), right=float(result[1]), mean=float(result.mean()),
                windows=len(x), members=54, seconds=time.perf_counter()-started,
                predictor=MANIFEST['predictor'], clinicalValidation=False,
                modelManifestSHA256=sha(ASSETS/'manifest.json'), comparison=comparison)
