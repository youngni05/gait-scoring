"""Analyze supplied pose packets with the frozen research measurements."""
import hashlib
import json
import os
import platform
import importlib
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "2")
HERE = Path(__file__).resolve().parent
import numpy as np
import scipy
from schema import ANALYSIS_VERSION, LIMITS, RequestError, validate
from vendor import knee, gait_events
from segmentation import detect as detect_segments
from current_report import build_report, foot_tracking


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


LINEAGE = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
for source in LINEAGE["sources"]:
    if sha(HERE / source["vendored"]) != source["vendored_sha256"]:
        raise RuntimeError("Vendored measurement source hash mismatch")
POSE_SPEC = json.loads((HERE / "pose-contract.json").read_text(encoding="utf-8"))
MP_INDICES = POSE_SPEC["output"]["mediapipe_indices"]
JOINT_ORDER = POSE_SPEC["output"]["joint_order"]
SOURCE_PATHS = ("core.py", "schema.py", "segmentation.py", "current_report.py", "contract.json", "pose-contract.json", "sources.json",
                "vendor/knee.py", "vendor/gait_events.py", "vendor/io_support.py")
LOADED_SOURCE_SHA256 = {name: sha(HERE / name) for name in SOURCE_PATHS}
GDI_MODULE = None
GDI_SOURCE_SHA256 = None


def execution():
    return dict(python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__,
                source_sha256=dict(LOADED_SOURCE_SHA256, **({"gdi_adapter.py": GDI_SOURCE_SHA256} if GDI_SOURCE_SHA256 else {})),
                measurement_methods=["knee-flexion-2d/1", gait_events.METHOD])


def health():
    return dict(schema="ai-health/1", status="ok", analysis_version=ANALYSIS_VERSION, limits=LIMITS)


def mapped_pose(pose):
    frames = []
    for f in pose["frames"]:
        joints = None
        if f["landmarks"] is not None:
            joints = []
            for i in MP_INDICES:
                x, y, _, visibility, presence = f["landmarks"][i]
                xy = [x * pose["image"]["width"], y * pose["image"]["height"]] if x is not None and y is not None else [None, None]
                joints.append([*xy, visibility, presence])
        frames.append(dict(index=f["index"], source_time_ms=f["source_time_ms"],
                           why_unavailable=f["why_unavailable"], joints=joints))
    return dict(schema="measurement-joints/1", image=pose["image"], clock=pose["clock"],
                source_id=pose["source_id"], joint_order=JOINT_ORDER, frames=frames)


def comparison(request):
    pose, provenance = request["pose"], request["provenance"]
    unknown = []
    for key, value in pose["extractor"].items():
        if value is None or value == "unknown":
            unknown.append("extractor." + key)
    required_settings = ("runningMode", "numPoses", "minPoseDetectionConfidence",
                         "minPosePresenceConfidence", "minTrackingConfidence", "outputSegmentationMasks")
    if pose["extractor"]["name"] == "OpenPose":
        required_settings = ("net_height", "person_selection", "coordinate_layout")
    for key in required_settings:
        if pose["extractor"]["settings"].get(key) in (None, "unknown"):
            unknown.append("extractor.settings." + key)
    if provenance["input_object"] == "unknown":
        unknown.append("input_object")
    if pose["clock"]["basis"] == "unknown":
        unknown.append("clock.basis")
    for key, value in provenance["tracking"].items():
        if value is None or value == "unknown":
            unknown.append("tracking." + key)
    for i, segment in enumerate(request["segments"]):
        if segment["direction"] == "unknown":
            unknown.append(f"segments[{i}].direction")
    track = {k: v for k, v in provenance["tracking"].items() if k != "session_id"}
    return dict(ready=not unknown, unknown_fields=unknown, metadata_basis="reported",
                key=dict(analysis_version=ANALYSIS_VERSION, extractor=pose["extractor"],
                         image={k: pose["image"][k] for k in ("width", "height", "orientation")},
                         clock=pose["clock"], input_object=provenance["input_object"], tracking=track))


def knee_result(pose, frames):
    raw = np.array([f["landmarks"] if f["landmarks"] is not None else [[None] * 5] * 33 for f in frames], dtype=float)
    raw = raw.reshape(len(frames), 33, 5)
    clocks = [f["source_time_ms"] for f in frames]
    motion = knee.measure(raw, clocks, pose["image"]["width"], pose["image"]["height"])
    metrics = {}
    spans = {}
    for view in knee.VIEWS:
        summary = motion["views"][view]
        def metric(value, count):
            return dict(value=value, unit="deg", valid_count=count,
                        unavailable_reason=None if value is not None else "fewer_than_two_valid_frames")
        metrics[view + "_knee_range"] = {
            side: metric(summary[side]["range_deg"], summary[side]["valid_frames"]) for side in ("left", "right")}
        common = summary["bilateral"]
        metrics[view + "_left_minus_right_range"] = metric(common["left_minus_right_range_deg"], common["common_valid_frames"])
        spans[view] = {}
        for side in ("left", "right"):
            valid_times = [t for t, angle in zip(clocks, summary["curve_deg"][side]) if angle is not None]
            spans[view][side] = valid_times[-1] - valid_times[0] if len(valid_times) >= 2 else 0
    return dict(method_id="knee-flexion-2d/1", metrics=metrics, quality=motion["quality"],
                valid_span_ms=spans, views=motion["views"])


def unavailable_gait(pose, segment, frame_count):
    reason = "unknown_direction"
    scalar = lambda unit: dict(value=None, unit=unit, valid_count=0, unavailable_reason=reason)
    return dict(schema="measurement-metrics/1", source_id=pose["source_id"], start_ms=segment["start_ms"],
                stop_ms=segment["stop_ms"], method_id=gait_events.METHOD, event_kind="pose_estimate",
                source_frames=frame_count, quality_reasons=[reason], events=[], steps=[], rejected_pairs={},
                metrics=dict(cadence=scalar("steps/min"), valid_steps=scalar("count"),
                             step_time={side: scalar("s") for side in ("left", "right")},
                             left_minus_right_step_time=scalar("s")))


def metric_values(metrics):
    for value in metrics.values():
        if "value" in value:
            yield value["value"]
        else:
            yield from metric_values(value)


def result_status(knees, gait, gdi=None):
    values = list(metric_values(knees["metrics"])) + list(metric_values(gait["metrics"]))
    if gdi is not None:
        for name in ("baseline", "normal100"):
            model = gdi.get("models", {}).get(name, {})
            for side in ("left", "right"):
                values.append(model.get("value", {}).get(side) if model.get("status") == "available" else None)
    available = sum(value is not None for value in values)
    return "complete" if available == len(values) else "partial" if available else "unavailable"


def gdi_unavailable(reason, status="unavailable"):
    return dict(schema="gdi-segment/1", status=status, unavailable_reason=reason,
        models={name: dict(status=status, value=dict(left=None, right=None), unit="GDI points",
                           unavailable_reason=reason) for name in ("baseline", "normal100")})


def analyze_gdi(pose, segment):
    global GDI_MODULE, GDI_SOURCE_SHA256
    path = HERE / "gdi_adapter.py"
    if not path.is_file():
        return gdi_unavailable("gdi_adapter_not_connected", "not_configured")
    try:
        if GDI_MODULE is None:
            module = importlib.import_module("gdi_adapter")
            if Path(module.__file__).resolve() != path.resolve():
                return gdi_unavailable("gdi_adapter_path_mismatch")
            GDI_MODULE, GDI_SOURCE_SHA256 = module, sha(path)
        return GDI_MODULE.analyze_segment(pose, segment)
    except ImportError:
        return gdi_unavailable("gdi_adapter_dependency_unavailable")
    except Exception as error:
        result = gdi_unavailable("gdi_inference_failed")
        result["error_type"] = type(error).__name__
        return result


def _summarize_direction(segments):
    accepted, excluded = [], []
    intervals = []
    for segment in segments:
        low, high = segment["start_ms"], segment["stop_ms"]
        if any(low < b and a < high for a, b in intervals):
            excluded.append(dict(segment_id=segment["segment_id"], reason="overlapping_segment"))
            continue
        accepted.append(segment)
        intervals.append((low, high))
    models = {}
    for name in ("baseline", "normal100"):
        values = []
        omitted = list(excluded)
        for segment in accepted:
            model = segment["gdi"].get("models", {}).get(name, {})
            value = model.get("value", {})
            if model.get("status") == "available" and all(
                type(value.get(side)) in (float, int) and np.isfinite(value[side]) for side in ("left", "right")):
                values.append((segment["segment_id"], value))
            else:
                omitted.append(dict(segment_id=segment["segment_id"], reason=model.get("unavailable_reason") or "gdi_unavailable"))
        models[name] = dict(status="available" if values else "unavailable",
            value={side: float(np.mean([v[side] for _, v in values])) if values else None for side in ("left", "right")},
            unit="GDI points", aggregation="equal_nonoverlapping_segment_mean",
            used_segment_ids=[identity for identity, _ in values], excluded_segments=omitted,
            unavailable_reason=None if values else "no_available_segment_gdi")
    count = sum(model["status"] == "available" for model in models.values())
    return dict(schema="gdi-summary/1", status="available" if count == 2 else "partial" if count else "unavailable",
                models=models, segment_count=len(segments))


def summarize_gdi(segments):
    directions = sorted({segment["direction"] for segment in segments})
    grouped = {direction: _summarize_direction([s for s in segments if s["direction"] == direction])
               for direction in directions}
    if len(directions) == 1 and directions[0] != "unknown":
        return dict(grouped[directions[0]], by_direction=grouped, aggregation_scope="single_direction")
    reason = "mixed_directions_not_pooled" if len(directions) > 1 else "unknown_direction_not_pooled" if directions else "no_detected_segments"
    has_value = any(row["status"] in ("available", "partial") for row in grouped.values())
    return dict(schema="gdi-summary/1", status="partial" if has_value else "unavailable",
        models={name: dict(status="unavailable", value=dict(left=None, right=None), unit="GDI points",
                           unavailable_reason=reason, used_segment_ids=[]) for name in ("baseline", "normal100")},
        segment_count=len(segments), by_direction=grouped, aggregation_scope="by_direction", unavailable_reason=reason)


def analyze(request, *, review_contacts=False):
    validate(request)
    v2 = request["schema"] == "ai-analyze-request/2"
    pose = request["pose"]
    automatic = "segments" not in request
    if automatic:
        detection = detect_segments(pose, LIMITS["max_segments"])
        resolved = [{key: row[key] for key in ("segment_id", "start_ms", "stop_ms", "direction")}
                    for row in detection["segments"]]
    else:
        resolved = request["segments"]
        detection = dict(mode="explicit", method_id="caller-specified/1", status="supplied",
                         segments=resolved, excluded=[], boundaries=[])
    effective = dict(request, segments=resolved)
    mapped = mapped_pose(pose)
    segments, frame_groups = [], []
    for segment in resolved:
        start, stop = segment["start_ms"], segment["stop_ms"]
        selected = [f for f in pose["frames"] if start <= f["source_time_ms"] < stop]
        knees = knee_result(pose, selected)
        if segment["direction"] == "unknown":
            gait = unavailable_gait(pose, segment, len(selected))
        elif review_contacts:
            from contact_review import measure_contacts
            tracking = foot_tracking(selected, pose["image"], [])
            gait = measure_contacts(mapped, start, stop, segment["direction"].removeprefix("image_"), tracking)
        else:
            gait = gait_events.measure_payload(mapped, start, stop, segment["direction"].removeprefix("image_"))
        stamps = [f["source_time_ms"] for f in selected]
        observed = [f["source_time_ms"] for f in selected if f["landmarks"] is not None]
        prior = sum(f["source_time_ms"] < start for f in pose["frames"])
        segments.append(dict(**segment, status=result_status(knees, gait),
            observation=dict(source_frame_count=len(selected), pose_frame_count=len(observed),
                observed_span_ms=stamps[-1] - stamps[0] if len(stamps) >= 2 else 0,
                pose_span_ms=observed[-1] - observed[0] if len(observed) >= 2 else 0,
                first_source_ms=stamps[0] if stamps else None, last_source_ms=stamps[-1] if stamps else None,
                source_indices=[f["index"] for f in selected], provided_prior_frame_count=prior),
            comparison=dict(direction=segment["direction"], sides="anatomical", provided_prior_frame_count=prior),
            knee=knees, gait=gait, validation=dict(status="experimental", pose_reference="unavailable", contact_reference="unavailable")))
        if v2:
            frame_groups.append(selected)
            segments[-1]["gdi"] = analyze_gdi(pose, segment)
            segments[-1]["status"] = result_status(knees, gait, segments[-1]["gdi"])
    states = [s["status"] for s in segments]
    status = "unavailable" if not states or all(s == "unavailable" for s in states) else "complete" if all(s == "complete" for s in states) else "partial"
    response = dict(schema="ai-analysis/2" if v2 else "ai-analysis/1", analysis_version=ANALYSIS_VERSION, request_id=request["request_id"],
        source_id=pose["source_id"], status=status, execution=execution(),
        provenance=dict(image=pose["image"], clock=pose["clock"], extractor=pose["extractor"], **request["provenance"]),
        comparison=comparison(effective), segments=segments,
        gdi=summarize_gdi(segments) if v2 else dict(status="not_configured", value=None, unavailable_reason="gdi_model_not_connected"))
    if review_contacts:
        from contact_review import METHOD
        response["execution"]["measurement_methods"].append(METHOD)
        response["execution"]["source_sha256"]["contact_review.py"] = sha(HERE / "contact_review.py")
    if v2:
        response["segmentation"] = detection
        response["current_report"] = build_report(pose["source_id"], segments, frame_groups, pose["image"], detection)
    # Guarantees strict serialization for every normal entry point.
    json.dumps(response, allow_nan=False)
    return response
