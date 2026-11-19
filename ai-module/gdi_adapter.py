"""Run both original frozen CNN/STT/bilateral ensembles on one native124 crop."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_LOCK = threading.RLock()
_RUNTIME = None
_MODELS = {}
_PINNED = {
    "studies/20261117-test02-fusion/protocol.json": "6f73b213e3d357246820f0e821577ec1905bdc66a17460bc39c8ed6490a2ddd9",
    "outputs/github-research-20261106/research/selected/frozen-gdi/models.json": "0b7d97e19ed5b5e06341828c4b20ee610527d32398063f73b01984b43464a6e9",
    "outputs/github-research-20261106/research/paper-related/normal100-models/models.json": "3a4a979697acb4412c13a43ecf567145804a188d7ef6b231b52a610a1a700457",
    "studies/20261117-test02-fusion/study.py": "20678d19da073377c4a4e212e3c60ea8e4d325b7fbce4a1f1b3c189d94367252",
    "studies/20261117-shortwalk-frozen/frozen_core.py": "b35efceb70432033a02308e9a100eb8fdc12e3800af161ebdbb01ff396eda5f5",
    "outputs/team-share-gdi-normal-cnn-20261031-v2/bundle/preprocess.py": "3e20b68ae4396e96401ad167e07650ab881efa860146c54f0eefabf2e9e1471c",
    "outputs/team-share-gdi-normal-cnn-20261031-v2/bundle/sources/video_process_utils.py": "0dab14d35b2bc3eaabad8901c7d02a9ec08fa2bb8d3077a2866b57579a6ae1ac",
    "outputs/team-share-gdi-normal-cnn-20261031-v2/bundle/sources/process_raw_videos.ipynb": "34188bf66081858b2395b173375b87f510543a5d8d6ff7093bcd52299d8b1b1e",
}
MAP = {9:24,10:26,11:28,12:23,13:25,14:27,19:31,22:32}


class GdiUnavailable(Exception):
    def __init__(self, reason):
        self.reason = reason


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    # Same checked Python 3.12 dependencies used by the API, including scipy.
    dependency = ROOT / "gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps"
    if dependency.is_dir() and str(dependency) not in sys.path:
        sys.path.append(str(dependency))
    for relative, expected in _PINNED.items():
        if sha(ROOT/relative) != expected:
            raise GdiUnavailable("gdi_source_hash_mismatch")
    # Reuse original, immutable inference code and server-owned weights.
    fusion = load_module(ROOT/"studies/20261117-test02-fusion/study.py", "integration_original_gdi_fusion")
    configured = os.environ.get("AI_GDI_WEIGHTS_ROOT")
    if configured:
        fusion.MIX = Path(configured).resolve()
    protocol = fusion.read(ROOT/"studies/20261117-test02-fusion/protocol.json")
    for path, expected in protocol["source_sha256"].items():
        if sha(Path(path)) != expected:
            raise GdiUnavailable("gdi_source_hash_mismatch")
    try:
        members = fusion.definitions()
    except AssertionError as error:
        # The pinned original reports a missing checkpoint as AssertionError(Path).
        # Preserve every other assertion as a configuration/inference failure.
        if len(error.args) == 1 and isinstance(error.args[0], Path) and not error.args[0].is_file():
            raise FileNotFoundError(error.args[0]) from error
        raise
    assert len(members) == 109
    assert sum(m["arm"]=="baseline" for m in members) == 54
    assert sum(m["arm"]=="normal100" for m in members) == 55
    by_identity = {(m["arm"],m["family"],m["fold"],m["tag"]):m for m in protocol["members"]}
    for member in members:
        expected = by_identity[member["arm"],member["family"],member["fold"],member["tag"]]
        if member["source_sha256"] != expected["source_sha256"] or member["applied_calibration"] != expected["applied_calibration"]:
            raise GdiUnavailable("gdi_manifest_mismatch")
    prep = load_module(ROOT/"outputs/team-share-gdi-normal-cnn-20261031-v2/bundle/preprocess.py", "integration_original_gdi_preprocess")
    selections = fusion.read(fusion.PUBLIC/"models.json")["selections"]
    _RUNTIME = dict(fusion=fusion,prep=prep,ns=prep.source_functions(),members=members,selections=selections)
    return _RUNTIME


def prepare_features(pose, segment):
    context = runtime()
    np = context["fusion"].np
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
    ns,prep = context["ns"],context["prep"]
    missing = float(ns["max_pct_nan_or_zero_given_cols"](ns["drop_confidence_cols"](raw),prep.REQUIRED))
    if missing > .25:
        raise GdiUnavailable("gdi_required_coordinate_missing_over_25pct")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        processed = ns["process_video_and_add_cols"](raw,swap_orientation=False)
    x = np.stack([processed[:,prep.LEFT],processed[:,prep.RIGHT]]).astype(np.float32)[None]
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
                representation="MediaPipe xy/visibility to BODY25 approximate slots, crop-first original processing")
    return x,info


def infer_features(x):
    context=runtime()
    fusion,np = context["fusion"],context["fusion"].np
    raw={}
    for member in context["members"]:
        key=fusion.key(member)
        cache=_MODELS.get(key)
        if cache is None:
            if sha(Path(member["path"])) != member["source_sha256"]:
                raise GdiUnavailable("gdi_checkpoint_hash_mismatch")
            net,checkpoint=fusion.model_for(member)
            cache=(net,dict(lo=float(checkpoint["lo"]),span=float(checkpoint["span"])))
            _MODELS[key]=cache
        raw[key]=fusion.forward(cache[0],member["family"],x,cache[1])
    result={}
    for arm in ("baseline","normal100"):
        members=[m for m in context["members"] if m["arm"]==arm]
        prediction=fusion.aggregate(raw,members,context["selections"],np.array([0]),1)["fusion"].mean(0)[0]
        assert np.isfinite(prediction).all()
        result[arm]=dict(status="available",value=dict(left=float(prediction[0]),right=float(prediction[1])),
                         unit="GDI points",unavailable_reason=None,provenance=dict(
                             member_count=len(members),families=dict(CNN=sum(m["family"]=="CNN" for m in members),
                             STT=sum(m["family"]=="STT" for m in members),bilateral=sum(m["family"]=="bilateral" for m in members)),
                             calibration="original_training" if arm=="baseline" else "Stanford_training_only",
                             checkpoint_sha256={fusion.key(m):m["source_sha256"] for m in members},
                             composition="eligible CNN/STT bag, bilateral refit difference, equal five-fold mean",
                             selections=context["selections"]))
    return result,raw


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
            models,_=infer_features(x)
        except GdiUnavailable as error:
            return unavailable(error.reason,started)
        except FileNotFoundError:
            return unavailable("gdi_model_or_support_file_missing",started)
        except ImportError:
            return unavailable("gdi_runtime_dependency_unavailable",started)
        except (AssertionError,RuntimeError,ValueError,KeyError,TypeError):
            return unavailable("gdi_inference_or_configuration_failed",started)
    return dict(schema="gdi-segment/1",status="available",models=models,input=info,
                provenance=dict(method="full-ensemble-native124/1",device="CPU",model_cache_entries=len(_MODELS),
                                code_sha256=sha(Path(__file__)),source_sha256=_PINNED,
                                elapsed_seconds=time.perf_counter()-started,
                                validation="experimental; matched clinical GDI reference unavailable"))
