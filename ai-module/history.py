"""Deterministic, provenance-preserving comparisons of supplied analyze/2 visits."""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from math import isfinite
from pathlib import Path
import re

from schema import fields, finite, require, strict_value, text

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "history-contract.json").read_text(encoding="utf-8"))
POLICY = CONTRACT["comparison_policy_version"]
LIMITS = CONTRACT["limits"]
UNKNOWN = "unknown"
DATE_ZONE = timezone(timedelta(hours=9))
HASH = re.compile(r"[0-9a-f]{64}")
KNEE = ("observed_knee_range", "supported_knee_range", "observed_left_minus_right_range", "supported_left_minus_right_range")
GAIT = ("cadence", "valid_steps", "step_time", "left_minus_right_step_time")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


LOADED_SOURCES = {name: file_sha(name) for name in ("history.py", "history-contract.json")}


def known(value):
    return value is not None and value != UNKNOWN and value != ""


def capture(value, path):
    if value is None:
        return None
    text(value, path)
    require("T" in value and re.search(r"(?:Z|[+-]\d\d:\d\d)$", value) is not None,
            "Expected ISO8601 capture time with timezone or null", path)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(result.tzinfo is not None and result.utcoffset() is not None, "Capture timezone required", path)
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        require(False, "Invalid capture time", path)


def metric_shape(value, path):
    fields(value, ("value", "unit", "valid_count", "unavailable_reason"), path)
    require(value["value"] is None or finite(value["value"]), "Expected finite metric or null", path+".value")
    text(value["unit"], path+".unit")
    require(type(value["valid_count"]) is int and value["valid_count"] >= 0, "Invalid valid_count", path+".valid_count")
    require(value["unavailable_reason"] is None or isinstance(value["unavailable_reason"], str), "Invalid unavailable reason", path)


def metadata_texts(value, names, path):
    for name in names:
        if name in value and value[name] is not None:
            text(value[name], path+"."+name)


def hash_map(value, path):
    require(isinstance(value, dict), "Expected source hash object", path)
    require(all(isinstance(v, str) and (v == UNKNOWN or HASH.fullmatch(v) is not None) for v in value.values()),
            "Expected SHA256 values or unknown", path)


def validate_analysis(value, path):
    fields(value, ("schema", "analysis_version", "request_id", "source_id", "status", "execution", "provenance", "comparison", "segments", "gdi", "segmentation"), path)
    require(value["schema"] == "ai-analysis/2", "Expected unmodified ai-analysis/2", path+".schema")
    for key in ("analysis_version", "request_id", "source_id"):
        text(value[key], path+"."+key)
    require(value["status"] in ("complete", "partial", "unavailable"), "Invalid analysis status", path+".status")
    for key in ("execution", "provenance", "comparison", "gdi", "segmentation"):
        require(isinstance(value[key], dict), "Expected analysis object", path+"."+key)
    fields(value["provenance"], (), path+".provenance", ("image", "clock", "extractor", "input_object", "tracking"))
    for key in ("image", "clock", "extractor", "tracking"):
        require(key not in value["provenance"] or isinstance(value["provenance"][key], dict), "Expected metadata object", path+".provenance."+key)
    ext = value["provenance"].get("extractor", {})
    metadata_texts(ext, ("name", "variant", "runtime", "runtime_version", "model_sha256", "delegate"), path+".provenance.extractor")
    require(isinstance(ext.get("settings", {}), dict), "Expected extractor settings object", path+".provenance.extractor.settings")
    hash_map(value["execution"].get("source_sha256", {}), path+".execution.source_sha256")
    metadata_texts(value["provenance"], ("input_object",), path+".provenance")
    for name in ("image", "clock"):
        data = value["provenance"].get(name, {})
        metadata_texts(data, ("orientation",) if name == "image" else ("unit", "basis"), path+".provenance."+name)
        for dimension in ("width", "height") if name == "image" else ():
            require(dimension not in data or data[dimension] in (None, UNKNOWN) or type(data[dimension]) is int and data[dimension] > 0,
                    "Invalid image dimension metadata", path+".provenance.image."+dimension)
    detection = value["segmentation"]
    metadata_texts(detection, ("mode", "method_id"), path+".segmentation")
    require("settings" not in detection or isinstance(detection["settings"], dict), "Invalid segmentation settings", path+".segmentation.settings")
    segments = value["segments"]
    require(isinstance(segments, list) and len(segments) <= LIMITS["max_segments_per_analysis"], "Invalid analysis segment count", path+".segments")
    seen = set()
    for i, segment in enumerate(segments):
        p = path+f".segments[{i}]"
        fields(segment, ("segment_id", "start_ms", "stop_ms", "direction", "status", "observation", "comparison", "knee", "gait", "validation", "gdi"), p)
        text(segment["segment_id"], p+".segment_id")
        require(segment["segment_id"] not in seen, "Duplicate segment_id", p)
        seen.add(segment["segment_id"])
        require(finite(segment["start_ms"]) and finite(segment["stop_ms"]) and segment["start_ms"] < segment["stop_ms"], "Invalid segment bounds", p)
        require(segment["direction"] in ("image_left", "image_right", "unknown"), "Invalid direction", p+".direction")
        require(segment["status"] in ("complete", "partial", "unavailable"), "Invalid segment status", p+".status")
        for key in ("observation", "comparison", "knee", "gait", "validation", "gdi"):
            require(isinstance(segment[key], dict), "Expected segment object", p+"."+key)
        for area, names in (("knee", KNEE), ("gait", GAIT)):
            if "method_id" in segment[area]:
                text(segment[area]["method_id"], p+"."+area+".method_id")
            metric = segment[area].get("metrics", {})
            fields(metric, (), p+"."+area+".metrics", names)
            for name, entry in metric.items():
                if name in ("observed_knee_range", "supported_knee_range", "step_time"):
                    fields(entry, ("left", "right"), p+"."+area+"."+name)
                    for side in ("left", "right"):
                        metric_shape(entry[side], p+"."+area+"."+name+"."+side)
                else:
                    metric_shape(entry, p+"."+area+"."+name)
        models = segment["gdi"].get("models", {})
        require(segment["gdi"].get("input") is None or isinstance(segment["gdi"].get("input"), dict), "Invalid GDI input provenance", p+".gdi.input")
        require(isinstance(segment["gdi"].get("provenance", {}), dict), "Invalid GDI provenance", p+".gdi.provenance")
        gp = segment["gdi"].get("provenance", {})
        metadata_texts(gp, ("method", "code_sha256"), p+".gdi.provenance")
        if "source_sha256" in gp:
            hash_map(gp["source_sha256"], p+".gdi.provenance.source_sha256")
        inp = segment["gdi"].get("input") or {}
        require(isinstance(inp.get("sampling", {}), dict), "Invalid sampling provenance", p+".gdi.input.sampling")
        metadata_texts(inp, ("selection",), p+".gdi.input")
        if "selected_frames" in inp:
            require(type(inp["selected_frames"]) is int and inp["selected_frames"] >= 0, "Invalid selected_frames", p+".gdi.input.selected_frames")
        sampling = inp.get("sampling", {})
        metadata_texts(sampling, ("orientation_processing", "training_basis"), p+".gdi.input.sampling")
        require("time_resampling" not in sampling or sampling["time_resampling"] is None or type(sampling["time_resampling"]) is bool,
                "Invalid time resampling flag", p+".gdi.input.sampling.time_resampling")
        fields(models, (), p+".gdi.models", ("baseline", "normal100"))
        for name, model in models.items():
            fields(model, ("status", "value", "unit", "unavailable_reason"), p+".gdi.models."+name, ("provenance",))
            require(model["status"] in ("available", "unavailable", "not_configured"), "Invalid GDI status", p)
            fields(model["value"], ("left", "right"), p+".gdi.models."+name+".value")
            require(all(v is None or finite(v) for v in model["value"].values()), "Invalid GDI value", p)
            text(model["unit"], p+".gdi.models."+name+".unit")
            require(isinstance(model.get("provenance", {}), dict), "Invalid model provenance", p+".gdi.models."+name+".provenance")
            mp = model.get("provenance", {})
            if "checkpoint_sha256" in mp:
                hash_map(mp["checkpoint_sha256"], p+".gdi.models."+name+".provenance.checkpoint_sha256")
            metadata_texts(mp, ("calibration", "composition"), p+".gdi.models."+name+".provenance")
    return value


def validate(request):
    strict_value(request)
    fields(request, ("schema", "request_id", "comparison_policy_version", "current_visit", "prior_visits"), "$", ("initial_reference_visit_id", "data_kind"))
    require(request["schema"] == CONTRACT["request_schema"], "Unsupported history schema", "$.schema")
    require(request["comparison_policy_version"] == POLICY, "Unsupported history policy", "$.comparison_policy_version")
    text(request["request_id"], "$.request_id")
    require(request.get("data_kind", "unspecified") in CONTRACT["request"]["data_kind"], "Invalid data_kind", "$.data_kind")
    if request.get("initial_reference_visit_id") is not None:
        text(request["initial_reference_visit_id"], "$.initial_reference_visit_id")
    require(isinstance(request["prior_visits"], list) and len(request["prior_visits"]) <= LIMITS["max_prior_visits"], "Invalid prior visit count", "$.prior_visits")
    visits, analyses = {}, {}
    total = 0
    for i, visit in enumerate([request["current_visit"], *request["prior_visits"]]):
        path = "$.current_visit" if i == 0 else f"$.prior_visits[{i-1}]"
        fields(visit, ("subject_id", "visit_id", "captured_at", "records"), path)
        for key in ("subject_id", "visit_id"):
            text(visit[key], path+"."+key)
        stamp = capture(visit["captured_at"], path+".captured_at")
        identity = (visit["subject_id"], stamp)
        require(visit["visit_id"] not in visits or visits[visit["visit_id"]] == identity, "Conflicting visit_id metadata", path+".visit_id")
        visits[visit["visit_id"]] = identity
        require(isinstance(visit["records"], list) and len(visit["records"]) <= LIMITS["max_records_per_visit"], "Invalid video record count", path+".records")
        total += len(visit["records"])
        for j, record in enumerate(visit["records"]):
            p = path+f".records[{j}]"
            fields(record, ("source_id", "source_sha256", "analysis_id", "analysis"), p, ("tracking_policy_id", "sampling_policy_id"))
            for key in ("source_id", "source_sha256", "analysis_id"):
                text(record[key], p+"."+key)
            require(record["source_sha256"] == UNKNOWN or HASH.fullmatch(record["source_sha256"]) is not None, "Invalid source SHA256", p+".source_sha256")
            for key in ("tracking_policy_id", "sampling_policy_id"):
                if key in record:
                    text(record[key], p+"."+key)
            validate_analysis(record["analysis"], p+".analysis")
            require(record["source_id"] == record["analysis"]["source_id"], "Source ID does not match analysis", p+".source_id")
            signature = canonical(record)
            require(record["analysis_id"] not in analyses or analyses[record["analysis_id"]] == signature, "Conflicting analysis_id payload", p+".analysis_id")
            analyses[record["analysis_id"]] = signature
    require(total <= LIMITS["max_total_records"], "Too many analysis records", "$")
    return request


def quantile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1)*p
    low = int(position)
    fraction = position-low
    value = ordered[low]*(1-fraction) + ordered[min(low+1, len(ordered)-1)]*fraction
    require(isfinite(value), "Derived statistic exceeds finite numeric range", "$")
    return float(value)


def spread(values):
    q1, q3 = quantile(values, .25), quantile(values, .75)
    return dict(median=quantile(values, .5), q1=q1, q3=q3, iqr=delta(q3, q1) if values else None)


def delta(a, b):
    value = float(a-b)
    require(isfinite(value), "Derived difference exceeds finite numeric range", "$")
    return value


def missing_fields(value, path=""):
    if isinstance(value, dict):
        if not value:
            return [path]
        return [field for key, item in value.items() for field in missing_fields(item, (path+"." if path else "")+key)]
    if isinstance(value, list):
        return [field for i, item in enumerate(value) for field in missing_fields(item, path+f"[{i}]")]
    return [path] if not known(value) else []


def condition(record, segment, area, method, unit, side):
    analysis = record["analysis"]
    prov = analysis["provenance"]
    ext, image, clock = (prov.get(key, {}) for key in ("extractor", "image", "clock"))
    ext_key = {key: ext.get(key, UNKNOWN) for key in ("name", "variant", "runtime", "runtime_version", "model_sha256", "delegate")}
    ext_key["settings"] = {key: ext.get("settings", {}).get(key, UNKNOWN) for key in
        ("runningMode", "numPoses", "minPoseDetectionConfidence", "minPosePresenceConfidence", "minTrackingConfidence", "outputSegmentationMasks")}
    # Preserve additional configured extractor options too.
    ext_key["settings"].update(ext.get("settings", {}))
    detection = analysis["segmentation"]
    key = dict(policy=POLICY, analysis_version=analysis["analysis_version"], method_id=method, unit=unit, side=side,
        direction=segment["direction"], extractor=ext_key,
        image={k: image.get(k, UNKNOWN) for k in ("width", "height", "orientation")},
        clock={k: clock.get(k, UNKNOWN) for k in ("unit", "basis")}, input_object=prov.get("input_object", UNKNOWN),
        tracking_policy_id=record.get("tracking_policy_id", UNKNOWN),
        segmentation=dict(mode=detection.get("mode", UNKNOWN), method_id=detection.get("method_id", UNKNOWN),
                          settings=detection.get("settings", "not_applicable" if detection.get("mode") == "explicit" else UNKNOWN)))
    hashes = analysis["execution"].get("source_sha256", {})
    if area in ("knee", "gait"):
        key["measurement_source_sha256"] = hashes.get("vendor/knee.py" if area == "knee" else "vendor/gait_events.py", UNKNOWN)
    else:
        model = segment["gdi"].get("models", {}).get(area, {})
        gdi_prov = segment["gdi"].get("provenance", {})
        model_prov = model.get("provenance", {})
        inp = segment["gdi"].get("input") or {}
        sampling = inp.get("sampling", {})
        key["gdi"] = dict(model=area, method=gdi_prov.get("method", UNKNOWN), code_sha256=gdi_prov.get("code_sha256", UNKNOWN),
            sampling_policy_id=record.get("sampling_policy_id", UNKNOWN),
            source_sha256=gdi_prov.get("source_sha256") or UNKNOWN,
            model_provenance={k: model_prov.get(k, UNKNOWN) for k in
                ("member_count", "families", "calibration", "checkpoint_sha256", "composition", "selections")},
            selection=inp.get("selection", UNKNOWN), selected_frames=inp.get("selected_frames", UNKNOWN),
            sampling={k: sampling.get(k, UNKNOWN) for k in ("time_resampling", "orientation_processing", "training_basis")})
    missing = missing_fields(key)
    if segment["direction"] == UNKNOWN:
        missing.append("direction")
    if record["source_sha256"] == UNKNOWN:
        missing.append("source_sha256")
    if record["source_id"] == UNKNOWN:
        missing.append("source_id")
    return key, sorted(set(missing))


def metric_entries(record):
    for segment in record["analysis"]["segments"]:
        for name in KNEE:
            sides = ("left", "right") if name.endswith("knee_range") else ("bilateral",)
            for side in sides:
                entry = segment["knee"].get("metrics", {}).get(name, {})
                if side != "bilateral":
                    entry = entry.get(side, {})
                yield segment, name, side, "knee", segment["knee"].get("method_id", UNKNOWN), entry
        for name in ("cadence", "step_time", "left_minus_right_step_time"):
            sides = ("left", "right") if name == "step_time" else ("none",) if name == "cadence" else ("bilateral",)
            for side in sides:
                entry = segment["gait"].get("metrics", {}).get(name, {})
                if name == "step_time":
                    entry = entry.get(side, {})
                yield segment, name, side, "gait", segment["gait"].get("method_id", UNKNOWN), entry
        for name in ("baseline", "normal100"):
            model = segment["gdi"].get("models", {}).get(name, {})
            for side in ("left", "right"):
                entry = dict(value=model.get("value", {}).get(side) if model.get("status") == "available" else None,
                    unit=model.get("unit", "GDI points"), valid_count=(segment["gdi"].get("input") or {}).get("selected_frames", 0),
                    unavailable_reason=model.get("unavailable_reason") or "gdi_unavailable")
                yield segment, "gdi_"+name, side, name, segment["gdi"].get("provenance", {}).get("method", UNKNOWN), entry


def normalize(request):
    current_id = request["current_visit"]["visit_id"]
    visits = {}
    for raw in [request["current_visit"], *request["prior_visits"]]:
        stamp = capture(raw["captured_at"], "$.captured_at")
        visit = visits.setdefault(raw["visit_id"], dict(visit_id=raw["visit_id"], subject_id=raw["subject_id"],
            time=stamp, captured_at=stamp.isoformat() if stamp else None, records={}))
        for record in raw["records"]:
            visit["records"][record["analysis_id"]] = record
    current = visits[current_id]
    ordered = sorted(visits.values(), key=lambda v: (v["time"] or datetime.min.replace(tzinfo=timezone.utc), v["visit_id"]))
    records = []
    for visit in ordered:
        for record in sorted(visit["records"].values(), key=lambda r: (r["analysis_id"], r["source_id"])):
            records.append((visit, record))
    # Connected components prevent chains of renamed/re-hashed aliases from counting twice.
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    identities = {}
    for i, (visit, record) in enumerate(records):
        is_current = visit["visit_id"] == current_id
        eligible_prior = (known(current["subject_id"]) and visit["subject_id"] == current["subject_id"]
                          and visit["time"] is not None and current["time"] is not None and visit["time"] < current["time"])
        if not is_current and not eligible_prior:
            continue
        tags = [("id", record["source_id"])] if known(record["source_id"]) else []
        if record["source_sha256"] != UNKNOWN:
            tags.append(("sha", record["source_sha256"]))
        for tag in tags:
            if tag in identities:
                parents[find(i)] = find(identities[tag])
            identities[tag] = i
    current_components = {find(i) for i, (v, _) in enumerate(records) if v["visit_id"] == current_id}
    items, excluded = [], []
    for i, (visit, record) in enumerate(records):
        reason = None
        if visit["visit_id"] != current_id:
            if not known(current["subject_id"]) or not known(visit["subject_id"]):
                reason = "insufficient_metadata"
            elif visit["subject_id"] != current["subject_id"]:
                reason = "different_subject"
            elif visit["time"] is None or current["time"] is None:
                reason = "insufficient_metadata"
            elif visit["time"] >= current["time"]:
                reason = "not_prior"
            elif find(i) in current_components:
                reason = "duplicate_source"
        base = dict(visit_id=visit["visit_id"], source_id=record["source_id"], analysis_id=record["analysis_id"])
        if reason:
            excluded.append(dict(base, reason=reason))
            continue
        for segment, metric_id, side, area, method, metric in metric_entries(record):
            unit = metric.get("unit", "deg" if area == "knee" else "s" if "time" in metric_id else "steps/min")
            key, missing = condition(record, segment, area, method, unit, side)
            if not known(visit["subject_id"]):
                missing.append("subject_id")
            if current["time"] is None:
                missing.append("captured_at")
            items.append(dict(**base, component=find(i), visit=visit, record=record, segment=segment,
                metric_id=metric_id, side=side, direction=segment["direction"], unit=unit, condition_key=key,
                condition_key_id=digest(key), missing_metadata=sorted(set(missing)), value=metric.get("value"),
                valid_count=metric.get("valid_count", 0), unavailable_reason=metric.get("unavailable_reason") or "metric_unavailable"))
    return current, visits, items, excluded


def exclusion(item, reason, **details):
    return dict(visit_id=item["visit_id"], source_id=item["source_id"], analysis_id=item["analysis_id"],
                segment_id=item["segment"]["segment_id"], reason=reason, **details)


def unique_exclusions(rows):
    return [json.loads(value) for value in sorted({canonical(row) for row in rows})]


def aggregate(items, excluded):
    by_component = defaultdict(list)
    for item in items:
        if item["value"] is None:
            excluded.append(exclusion(item, item["unavailable_reason"]))
        else:
            by_component[item["component"]].append(item)
    by_visit = defaultdict(list)
    for component in sorted(by_component):
        group = by_component[component]
        first = min(group, key=lambda x: (x["visit"]["time"] or datetime.min.replace(tzinfo=timezone.utc),
                                        x["visit_id"], x["analysis_id"], x["source_id"]))
        chosen = (first["visit_id"], first["analysis_id"], first["source_id"])
        chosen_rows = []
        for item in group:
            if (item["visit_id"], item["analysis_id"], item["source_id"]) != chosen:
                excluded.append(exclusion(item, "duplicate_source"))
            else:
                chosen_rows.append(item)
        accepted, intervals = [], []
        for item in sorted(chosen_rows, key=lambda x: (x["segment"]["start_ms"], x["segment"]["stop_ms"], x["segment"]["segment_id"])):
            low, high = item["segment"]["start_ms"], item["segment"]["stop_ms"]
            if any(low < b and a < high for a, b in intervals):
                excluded.append(exclusion(item, "overlapping_segment"))
            else:
                intervals.append((low, high))
                accepted.append(item)
        if not accepted:
            continue
        values = [r["value"] for r in accepted]
        by_visit[first["visit_id"]].append(dict(source_id=first["source_id"], source_sha256=first["record"]["source_sha256"],
            analysis_id=first["analysis_id"], value=quantile(values, .5), segment_count=len(accepted),
            valid_count=sum(r["valid_count"] for r in accepted), segment_ids=[r["segment"]["segment_id"] for r in accepted],
            segment_distribution=spread(values)))
    visits = {}
    for visit_id, sources in by_visit.items():
        sources.sort(key=lambda s: (s["source_id"], s["analysis_id"]))
        representative = next(i["visit"] for i in items if i["visit_id"] == visit_id)
        stats = spread([s["value"] for s in sources])
        visits[visit_id] = dict(visit_id=visit_id, captured_at=representative["captured_at"], value=stats["median"],
            source_count=len(sources), segment_count=sum(s["segment_count"] for s in sources), valid_count=sum(s["valid_count"] for s in sources),
            used_records=sources, source_iqr=stats["iqr"])
    return visits


def empty_visit(current):
    return dict(visit_id=current["visit_id"], captured_at=current["captured_at"], value=None, source_count=0,
                segment_count=0, valid_count=0, used_records=[], source_iqr=None)


def reference(summary, current_value, reason):
    if summary is None or current_value is None:
        return dict(value=None, delta=None, reference_visit_ids=[], captured_at=None, used_records=[], unavailable_reason=reason)
    return dict(value=summary["value"], delta=delta(current_value, summary["value"]), reference_visit_ids=[summary["visit_id"]],
                captured_at=summary["captured_at"], used_records=summary["used_records"], unavailable_reason=None)


def compare_history(request):
    validate(request)
    current, visits, items, global_excluded = normalize(request)
    groups = defaultdict(list)
    for item in items:
        if item["visit_id"] == current["visit_id"]:
            groups[(item["metric_id"], item["direction"], item["side"], item["condition_key_id"])].append(item)
    prepared = []
    for identity, current_items in sorted(groups.items()):
        sample = current_items[0]
        omitted = list(global_excluded)
        matching = []
        missing = sorted({key for i in current_items for key in i["missing_metadata"]})
        for item in items:
            if (item["metric_id"], item["direction"], item["side"]) != identity[:3]:
                continue
            if item["visit_id"] == current["visit_id"]:
                if item["condition_key_id"] == identity[3]:
                    matching.append(item)
            elif item["missing_metadata"]:
                omitted.append(exclusion(item, "insufficient_metadata", fields=item["missing_metadata"]))
            elif item["condition_key_id"] != identity[3]:
                omitted.append(exclusion(item, "condition_mismatch"))
            elif missing:
                omitted.append(exclusion(item, "insufficient_metadata", fields=missing))
            else:
                matching.append(item)
        summaries = aggregate(matching, omitted)
        current_summary = summaries.pop(current["visit_id"], empty_visit(current))
        if missing:
            summaries = {}
        prepared.append(dict(metric_id=sample["metric_id"], direction=sample["direction"], side=sample["side"], unit=sample["unit"],
            condition_key_id=sample["condition_key_id"], condition_key=sample["condition_key"], missing_metadata=missing,
            current=current_summary, past=summaries, excluded_records=unique_exclusions(omitted)))
    fixed = request.get("initial_reference_visit_id")
    usable_ids = {v for row in prepared if row["current"]["value"] is not None for v in row["past"]}
    proposal = min(usable_ids, key=lambda v: (visits[v]["time"], v)) if usable_ids else None
    if proposal is None and any(row["current"]["value"] is not None and not row["missing_metadata"] for row in prepared):
        proposal = current["visit_id"]
    effective = fixed if fixed is not None else proposal if proposal != current["visit_id"] else None
    initial_reason = None
    if effective is not None:
        if effective not in visits:
            initial_reason = "initial_reference_not_found"
        elif effective == current["visit_id"] or visits[effective]["time"] is not None and current["time"] is not None and visits[effective]["time"] >= current["time"]:
            initial_reason = "not_prior"
        elif visits[effective]["subject_id"] != current["subject_id"]:
            initial_reason = "different_subject"
        elif visits[effective]["time"] is None or current["time"] is None:
            initial_reason = "insufficient_metadata"
    else:
        initial_reason = "no_prior_record"
    results = []
    for row in prepared:
        past = row.pop("past")
        ordered = sorted(past.values(), key=lambda v: (visits[v["visit_id"]]["time"], v["visit_id"]))
        reason = "current_metric_unavailable" if row["current"]["value"] is None else "insufficient_metadata" if row["missing_metadata"] else "no_comparable_metric" if request["prior_visits"] else "no_prior_record"
        recent = min(ordered, key=lambda v: (-visits[v["visit_id"]]["time"].timestamp(), v["visit_id"])) if ordered else None
        initial = past.get(effective) if initial_reason is None else None
        row["recent"] = reference(recent, row["current"]["value"], reason)
        fixed_reason = initial_reason or next((e["reason"] for e in row["excluded_records"] if e["visit_id"] == effective), "no_comparable_metric")
        row["initial"] = reference(initial, row["current"]["value"], reason if row["current"]["value"] is None or row["missing_metadata"] else fixed_reason)
        distribution = spread([v["value"] for v in ordered])
        row["past_distribution"] = dict(distribution, n_visits=len(ordered), n_dates=len({visits[v["visit_id"]]["time"].astimezone(DATE_ZONE).date() for v in ordered}),
            reference_visit_ids=[v["visit_id"] for v in ordered])
        timeline = [dict(v, is_current=False) for v in ordered] + [dict(row["current"], is_current=True)]
        row["timeline"] = sorted(timeline, key=lambda v: (v["captured_at"] or "", v["visit_id"]))
        count = sum(row[name]["delta"] is not None for name in ("recent", "initial"))
        row["status"] = "complete" if count == 2 else "partial" if count else "unavailable"
        results.append(row)
    states = [row["status"] for row in results]
    status = "complete" if states and all(s == "complete" for s in states) else "unavailable" if not states or all(s == "unavailable" for s in states) else "partial"
    response = dict(schema=CONTRACT["response_schema"], request_id=request["request_id"], comparison_policy_version=POLICY,
        data_kind=request.get("data_kind", "unspecified"), status=status, current_visit_id=current["visit_id"],
        initial_reference=dict(requested_visit_id=fixed, effective_visit_id=effective if initial_reason is None else None,
            proposed_visit_id=proposal if fixed is None else None, requires_persistence=fixed is None and proposal is not None,
            unavailable_reason=initial_reason), metrics=results, excluded_records=unique_exclusions(global_excluded),
        execution=dict(source_sha256=LOADED_SOURCES, metadata_basis="reported", date_timezone="UTC+09:00", clinical_thresholds_configured=False))
    json.dumps(response, allow_nan=False)
    return response
