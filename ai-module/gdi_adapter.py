"""Run both original frozen CNN/STT/bilateral ensembles on one native124 crop."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading
import time
import warnings

import numpy as np

import preprocess
from prepare_assets import GDI_PACKAGES, STANFORD_FILES, sha, verify

ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.RLock()
_RUNTIME = None
_MANIFESTS = {
    "baseline": ("research/selected/frozen-gdi/models.json",
                 "0b7d97e19ed5b5e06341828c4b20ee610527d32398063f73b01984b43464a6e9", 54, "original"),
    "normal100": ("research/paper-related/normal100-models/models.json",
                  "3a4a979697acb4412c13a43ecf567145804a188d7ef6b231b52a610a1a700457", 55, "stanford"),
}
_STT_FILES = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "Attention.py": "b966aff7b28adc65eb9258d74e53d23bbaa1d2bec939a7124afbc17a89b45a48",
    "MLP.py": "1f589b03bfd87132055d903881958ab1b32a46e89d8a8a8abeff74de841ab31c",
    "Transformer.py": "35cbaef25129b14669161e6b7bf1a81494ab5c5ee3ebfe7aa73d1566867f39a0",
}
MAP = {9:24,10:26,11:28,12:23,13:25,14:27,19:31,22:32}


class GdiUnavailable(Exception):
    def __init__(self, reason):
        self.reason = reason


def configure(assets, stt_source):
    """Verify the released files before allowing the app to accept a video."""
    global _RUNTIME
    source_hashes = {}
    for filename, expected in _STT_FILES.items():
        source = stt_source / "model" / filename
        normalized = source.read_bytes().replace(bytes((13, 10)), bytes((10,)))
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise ValueError(f"STT source differs from the pinned commit: {source}")
        source_hashes[f"stt/model/{filename}"] = expected
    sources = assets / "stanford"
    for relative, expected in STANFORD_FILES.items():
        verify(sources / Path(relative).name, expected)
        source_hashes[f"stanford/{relative}"] = expected

    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str((ROOT / _MANIFESTS["baseline"][0]).parent))
    predictor = importlib.import_module("predict")
    arms = {}
    for arm, (relative, expected, count, calibration) in _MANIFESTS.items():
        manifest_path = ROOT / relative
        verify(manifest_path, expected)
        source_hashes[relative] = expected
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["members"] = [
            member for member in manifest["members"]
            if (member["tag"] == "refit" if member["family"] == "bilateral"
                else not member["excluded_from_bag"])
        ]
        if len(manifest["members"]) != count:
            raise ValueError(f"Unexpected number of model members for {arm}")
        weights = assets / GDI_PACKAGES[arm][0]
        checkpoints = []
        for member in manifest["members"]:
            path = weights / member["file"]
            verify(path, member["public_sha256"])
            saved_paths = sys.path[:]
            try:
                model = predictor.new_model(member["family"], stt_source)
            finally:
                sys.path[:] = saved_paths
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            checkpoints.append((member, model, float(checkpoint["lo"]), float(checkpoint["span"])))
        arms[arm] = (checkpoints, manifest["selections"], calibration)

    _RUNTIME = (predictor, arms, preprocess.source_functions(sources), source_hashes)


def runtime():
    if _RUNTIME is None:
        raise GdiUnavailable("gdi_model_or_support_file_missing")
    return _RUNTIME


def prepare_features(pose, segment):
    _, _, ns, _ = runtime()
    if pose["image"]["orientation"] != "upright_unmirrored":
        raise GdiUnavailable("gdi_unsupported_orientation")
    frames = [f for f in pose["frames"] if segment["start_ms"] <= f["source_time_ms"] < segment["stop_ms"]]
    if len(frames) < 124:
        raise GdiUnavailable("gdi_fewer_than_124_source_frames")
    start = (len(frames)-124)//2
    chosen = frames[start:start+124]
    ids = np.array([f["index"] for f in chosen])
    times = np.array([f["source_time_ms"] for f in chosen],float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0) or np.any(np.diff(ids)!=1):
        raise GdiUnavailable("gdi_noncontiguous_source_frames")
    delta = np.diff(times)
    if np.any(delta > 2*np.median(delta)):
        raise GdiUnavailable("gdi_source_time_gap")
    raw = np.full((124,25,3),np.nan)
    width,height = pose["image"]["width"],pose["image"]["height"]
    for i,frame in enumerate(chosen):
        points = frame["landmarks"]
        if points is None:
            continue
        for op,mp in MAP.items():
            p = points[mp]
            raw[i,op] = [float(p[0])*width if p[0] is not None else np.nan,
                         float(p[1])*height if p[1] is not None else np.nan,
                         float(p[3]) if p[3] is not None else np.nan]
    raw = raw.reshape(124,75)
    missing = float(ns["max_pct_nan_or_zero_given_cols"](ns["drop_confidence_cols"](raw),preprocess.REQUIRED))
    if missing > .25:
        raise GdiUnavailable("gdi_required_coordinate_missing_over_25pct")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        processed = ns["process_video_and_add_cols"](raw,swap_orientation=False)
    x = np.stack([processed[:,preprocess.LEFT],processed[:,preprocess.RIGHT]]).astype(np.float32)[None]
    if not np.isfinite(x).all():
        raise GdiUnavailable("gdi_nonfinite_features")
    info = dict(source_frames=len(frames),selected_frames=124,selection="central_contiguous_native124",
                source_indices=ids.tolist(),source_time_ms=times.tolist(),first_source_ms=float(times[0]),
                last_source_ms=float(times[-1]),required_missing_fraction=missing,
                segment_start_ms=segment["start_ms"],segment_stop_ms=segment["stop_ms"],
                crop_start_offset=start,crop_stop_offset_exclusive=start+124,
                sampling=dict(median_interval_ms=float(np.median(delta)),minimum_interval_ms=float(delta.min()),
                              maximum_interval_ms=float(delta.max()),span_ms=float(times[-1]-times[0]),
                              effective_sample_rate_hz=float(123000/(times[-1]-times[0])),
                              time_resampling=False,orientation_processing="swap_orientation=False",
                              training_basis="Original Stanford author-code nominal25Hz,124 frame slots; previous MediaPipe Test02 native about30Hz",
                              interpretation="Native124 retains observed PTS; sampling equivalence and clinical accuracy for this extractor are unvalidated"),
                feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                representation=("OpenPose anatomical xy/confidence from native BODY25, crop-first original processing" if pose["extractor"]["name"]=="OpenPose" else "MediaPipe xy/visibility to BODY25 approximate slots, crop-first original processing"))
    return x,info


def infer_features(x):
    predictor, arms, _, _ = runtime()
    models = {}
    for arm, (checkpoints, selections, calibration) in arms.items():
        members = [member for member, _, _, _ in checkpoints]
        calibration_key = "calibration" if calibration == "original" else "stanford_calibration"
        calibrated = {}
        for member, model, lo, span in checkpoints:
            raw = predictor.raw_forward(model, member["family"], x, lo, span)
            slope, intercept = member[calibration_key]
            key = (member["family"], member["fold"], member["tag"])
            calibrated[key] = raw.mean(axis=0, keepdims=True) * slope + intercept
        fold_results = []
        for fold, selection in enumerate(selections):
            cnn = np.mean([v for (family, i, _), v in calibrated.items() if family == "CNN" and i == fold], axis=0)
            stt = np.mean([v for (family, i, _), v in calibrated.items() if family == "STT" and i == fold], axis=0)
            bilateral = calibrated["bilateral", fold, "refit"]
            fold_results.append(predictor.combine(cnn, stt, bilateral, selection["cnn_weight"], selection["fraction"]))
        prediction = np.mean(fold_results, axis=0)[0]
        if not np.isfinite(prediction).all():
            raise GdiUnavailable("gdi_nonfinite_prediction")
        models[arm] = dict(
            status="available", value=dict(left=float(prediction[0]), right=float(prediction[1])),
            unit="GDI points", unavailable_reason=None,
            provenance=dict(
                member_count=len(members),
                families={family: sum(m["family"] == family for m in members)
                          for family in ("CNN", "STT", "bilateral")},
                calibration="original_training" if arm == "baseline" else "Stanford_training_only",
                checkpoint_sha256={
                    f"{arm}__{m['family']}__{m['fold']}__{m['tag']}": m["source_sha256"] for m in members
                },
                published_weight_sha256={m["file"]: m["public_sha256"] for m in members},
                composition="eligible CNN/STT bag, bilateral refit difference, equal five-fold mean",
                selections=selections,
            ),
        )
    return models


def unavailable(reason, started):
    return dict(schema="gdi-segment/1",status="unavailable",models={
        arm:dict(status="unavailable",value=dict(left=None,right=None),unit="GDI points",unavailable_reason=reason,
                 provenance=dict(expected_member_count=count))
        for arm,count in (("baseline",54),("normal100",55))},
        input=None,provenance=dict(method="full-ensemble-native124/1",elapsed_seconds=time.perf_counter()-started))


def analyze_segment(pose, segment):
    started=time.perf_counter()
    # Short requests do not import torch or load the ensemble.
    count=sum(segment["start_ms"]<=f["source_time_ms"]<segment["stop_ms"] for f in pose["frames"])
    if count<124:
        return unavailable("gdi_fewer_than_124_source_frames",started)
    with _LOCK:
        try:
            x,info=prepare_features(pose,segment)
            models=infer_features(x)
        except GdiUnavailable as error:
            return unavailable(error.reason,started)
        except FileNotFoundError:
            return unavailable("gdi_model_or_support_file_missing",started)
        except ImportError:
            return unavailable("gdi_runtime_dependency_unavailable",started)
        except (AssertionError,RuntimeError,ValueError,KeyError,TypeError):
            return unavailable("gdi_inference_or_configuration_failed",started)
    return dict(schema="gdi-segment/1",status="available",models=models,input=info,
                provenance=dict(method="full-ensemble-native124/1",device="CPU",
                                code_sha256=sha(Path(__file__)),source_sha256=runtime()[3],
                                elapsed_seconds=time.perf_counter()-started,
                                validation="experimental; matched clinical GDI reference unavailable"))
