"""Reuse the verified landmark adapter; isolate Lite inputs and outputs."""
import importlib.util
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
FULL=HERE.parent/'20261101-self-video-mediapipe'
spec=importlib.util.spec_from_file_location('full_common_reuse',FULL/'common.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
ROOT=shared.ROOT;OLD=shared.OLD;OLD_BIG=shared.OLD_BIG;MP_HOME=shared.MP_HOME
np=shared.np;read=shared.read;save=shared.save;sha=shared.sha;module=shared.module
adapt=shared.adapt;MAP=shared.MAP;JOINTS=shared.JOINTS
BIG=Path('D:/gait-research-data/D_P/self-video-mediapipe-lite-20261101')
FULL_BIG=shared.BIG
TASK=BIG/'pose_landmarker_lite.task'
FULL_TASK=shared.TASK
