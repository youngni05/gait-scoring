"""Explicit input validation and strict JSON handling for the local draft."""
import json
import math
import re
from pathlib import Path

CONTRACT = json.loads(Path(__file__).with_name("contract.json").read_text(encoding="utf-8"))
LIMITS = CONTRACT["limits"]
ANALYSIS_VERSION = CONTRACT["analysis_version"]


class RequestError(ValueError):
    def __init__(self, message, path=None, code="invalid_request", status=422):
        super().__init__(message)
        self.path, self.code, self.status = path, code, status

    def response(self):
        return {"schema": "ai-error/1", "error": {"code": self.code, "message": str(self), "path": self.path}}


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def require(condition, message, path):
    if not condition:
        raise RequestError(message, path)


def fields(value, required, path, optional=()):
    require(isinstance(value, dict), "Expected an object", path)
    require(set(required) <= value.keys(), "Missing required fields", path)
    require(value.keys() <= set(required) | set(optional), "Unknown fields", path)


def text(value, path):
    require(isinstance(value, str) and 0 < len(value) <= LIMITS["max_string_chars"], "Expected a nonempty bounded string", path)


def strict_value(value, path="$", depth=0):
    require(depth <= 20, "JSON nesting exceeds service limit", path)
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        require(math.isfinite(value), "Nonfinite numbers are invalid JSON", path)
    elif isinstance(value, (list, dict)):
        if isinstance(value, dict):
            require(all(isinstance(k, str) for k in value), "JSON keys must be strings", path)
            items = value.items()
        else:
            items = enumerate(value)
        for key, child in items:
            strict_value(child, f"{path}.{key}", depth + 1)
    else:
        raise RequestError("Expected strict JSON values", path)


def parse_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError("Invalid numeric constant")

    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeError, ValueError, RecursionError):
        raise RequestError("Expected valid UTF-8 strict JSON", code="invalid_json", status=400) from None
    return parsed


def validate(request):
    strict_value(request)
    fields(request, ("schema", "request_id", "pose", "provenance"), "$", ("segments",))
    require(request["schema"] in CONTRACT["request_schemas"], "Unsupported request schema", "$.schema")
    require(request["schema"] != "ai-analyze-request/1" or "segments" in request,
            "Version 1 requires explicit segments", "$.segments")
    text(request["request_id"], "$.request_id")
    pose = request["pose"]
    fields(pose, ("schema", "source_id", "image", "clock", "extractor", "frames"), "$.pose")
    require(pose["schema"] == "measurement-pose/1", "Unsupported pose schema", "$.pose.schema")
    text(pose["source_id"], "$.pose.source_id")
    image = pose["image"]
    fields(image, ("width", "height", "orientation", "preview_mirrored"), "$.pose.image")
    for key in ("width", "height"):
        require(type(image[key]) is int and 0 < image[key] <= LIMITS["max_image_dimension"], "Invalid pixel dimension", f"$.pose.image.{key}")
    require(image["orientation"] == "upright_unmirrored", "Expected upright unmirrored analysis image", "$.pose.image.orientation")
    require(image["preview_mirrored"] is None or type(image["preview_mirrored"]) is bool, "Expected boolean or null", "$.pose.image.preview_mirrored")
    clock = pose["clock"]
    fields(clock, ("unit", "basis"), "$.pose.clock")
    require(clock["unit"] == "ms", "Expected millisecond source times", "$.pose.clock.unit")
    require(clock["basis"] in ("video_pts", "capture_monotonic", "callback_monotonic", "unknown"), "Unsupported source clock", "$.pose.clock.basis")
    ext = pose["extractor"]
    fields(ext, ("name", "variant", "runtime", "runtime_version", "model_sha256", "delegate", "settings"), "$.pose.extractor")
    require(ext["name"] in ("MediaPipe Pose Landmarker","OpenPose"), "Unsupported landmark layout", "$.pose.extractor.name")
    require((ext["variant"] == "BODY_25" if ext["name"] == "OpenPose" else ext["variant"] in ("full", "lite", "unknown")), "Unsupported model variant", "$.pose.extractor.variant")
    require(ext["delegate"] in ("CPU", "GPU", "unknown"), "Unsupported delegate", "$.pose.extractor.delegate")
    for key in ("runtime", "model_sha256"):
        text(ext[key], f"$.pose.extractor.{key}")
    if ext["runtime_version"] is not None:
        text(ext["runtime_version"], "$.pose.extractor.runtime_version")
    require(ext["model_sha256"] == "unknown" or re.fullmatch(r"[0-9a-f]{64}", ext["model_sha256"]) is not None,
            "Expected model SHA256 or unknown", "$.pose.extractor.model_sha256")
    require(isinstance(ext["settings"], dict), "Expected extractor settings object", "$.pose.extractor.settings")
    frames = pose["frames"]
    require(isinstance(frames, list) and len(frames) <= LIMITS["max_frames"], "Frame limit exceeded or invalid array", "$.pose.frames")
    previous_index, previous_time = -1, -float("inf")
    for i, frame in enumerate(frames):
        path = f"$.pose.frames[{i}]"
        fields(frame, ("index", "source_time_ms", "landmarks", "why_unavailable"), path, ("detector_time_ms",))
        index, stamp = frame["index"], frame["source_time_ms"]
        require(type(index) is int and index > previous_index, "Source indices must strictly increase", path + ".index")
        require(finite(stamp) and stamp > previous_time, "Source times must strictly increase", path + ".source_time_ms")
        previous_index, previous_time = index, stamp
        require(frame.get("detector_time_ms") is None or finite(frame["detector_time_ms"]), "Invalid detector time", path + ".detector_time_ms")
        lm = frame["landmarks"]
        if lm is None:
            text(frame["why_unavailable"], path + ".why_unavailable")
            continue
        require(frame["why_unavailable"] is None, "Detected frames require null unavailability reason", path + ".why_unavailable")
        require(isinstance(lm, list) and len(lm) == 33, "Expected 33 landmarks", path + ".landmarks")
        for j, point in enumerate(lm):
            require(isinstance(point, list) and len(point) == 5 and all(v is None or finite(v) for v in point),
                    "Expected [x,y,z,visibility,presence] with finite values or null", path + f".landmarks[{j}]")
            if ext["name"]=="OpenPose":
                require((point[3] is None or point[3]>=0) and point[4] is None,
                        "Expected nonnegative native OpenPose response and null presence",path+f".landmarks[{j}]")
            else:
                require(all(v is None or 0 <= v <= 1 for v in point[3:]), "Visibility/presence must be within 0–1", path + f".landmarks[{j}]")
    segments = request.get("segments", [])
    require("segments" not in request or isinstance(segments, list) and 0 < len(segments) <= LIMITS["max_segments"],
            "Invalid segment count", "$.segments")
    seen = set()
    for i, segment in enumerate(segments):
        path = f"$.segments[{i}]"
        fields(segment, ("segment_id", "start_ms", "stop_ms", "direction"), path)
        text(segment["segment_id"], path + ".segment_id")
        require(segment["segment_id"] not in seen, "Duplicate segment_id", path + ".segment_id")
        seen.add(segment["segment_id"])
        require(finite(segment["start_ms"]) and finite(segment["stop_ms"]) and segment["start_ms"] < segment["stop_ms"],
                "Expected finite half-open segment bounds", path)
        require(segment["direction"] in ("image_left", "image_right", "unknown"), "Unsupported travel direction", path + ".direction")
    provenance = request["provenance"]
    fields(provenance, ("input_object", "tracking"), "$.provenance")
    require(provenance["input_object"] in ("ImageData", "HTMLVideoElement", "mp.Image", "BGR", "unknown"), "Unsupported input object", "$.provenance.input_object")
    track = provenance["tracking"]
    fields(track, ("session_id", "start_time_ms", "input_start_ms", "input_stop_ms", "prior_frame_count", "fresh_tracker"), "$.provenance.tracking")
    text(track["session_id"], "$.provenance.tracking.session_id")
    for key in ("start_time_ms", "input_start_ms", "input_stop_ms"):
        require(track[key] is None or finite(track[key]), "Expected finite time or null", "$.provenance.tracking." + key)
    count = track["prior_frame_count"]
    require(count is None or type(count) is int and count >= 0, "Invalid prior frame count", "$.provenance.tracking.prior_frame_count")
    require(track["fresh_tracker"] is None or type(track["fresh_tracker"]) is bool, "Expected boolean or null", "$.provenance.tracking.fresh_tracker")
    start, low, high = (track[k] for k in ("start_time_ms", "input_start_ms", "input_stop_ms"))
    require(start is None or low is None or start <= low, "Tracker start must precede input start", "$.provenance.tracking")
    require(low is None or high is None or low < high, "Invalid tracker input interval", "$.provenance.tracking")
    if frames:
        require(low is None or low <= frames[0]["source_time_ms"], "Input start excludes supplied frames", "$.provenance.tracking.input_start_ms")
        require(high is None or high > frames[-1]["source_time_ms"], "Input stop excludes supplied frames", "$.provenance.tracking.input_stop_ms")
    return request
