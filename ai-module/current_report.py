"""Explain the observations in one analyzed video using a fixed Korean policy."""
from statistics import mean

import numpy as np

from vendor.gait_events import calculate_steps


SCHEMA = "ai-current-report/1"
POLICY = "first-visit-observation/1"
SIDES = ("left", "right")
SIDE_LABELS = {"left": "왼쪽", "right": "오른쪽"}
SWITCH_FRACTION = .25
NEAR_CONTACT_MS = 75


def foot_tracking(frames, image, events):
    """Check adjacent source frames for a sharp switch-like heel movement."""
    clock = np.array([f["source_time_ms"] for f in frames])
    indices = np.array([f["index"] for f in frames])
    xy = np.array([f["landmarks"] if f["landmarks"] is not None else [[None] * 5] * 33
                   for f in frames], dtype=float).reshape(-1, 33, 5)
    size = np.array([image["width"], image["height"]])
    xy = xy[:, :, :2] * size
    hips, heels = xy[:, 23:25], xy[:, 29:31]
    lower = np.concatenate((hips, heels), axis=1)
    valid = np.isfinite(lower).all((1, 2)) & ((lower >= 0) & (lower <= size)).all((1, 2))
    leg = np.linalg.norm(hips - xy[:, 25:27], axis=2) + np.linalg.norm(xy[:, 25:27] - xy[:, 27:29], axis=2)
    leg = leg[np.isfinite(leg) & (leg > 0)]
    scale = float(np.median(leg)) if leg.size else None
    dt = np.diff(clock)
    checked = valid[:-1] & valid[1:] & (np.diff(indices) == 1)
    if dt.size:
        checked &= dt <= 2 * np.median(dt)
    transitions = []
    if scale is None:
        checked[:] = False
    elif np.any(checked):
        relative = heels - hips.mean(axis=1)[:, None]
        previous, current = relative[:-1][checked], relative[1:][checked]
        same = np.linalg.norm(current - previous, axis=2)
        exchanged = np.linalg.norm(current[:, ::-1] - previous, axis=2).sum(axis=1)
        threshold = SWITCH_FRACTION * scale
        flags = (same.sum(axis=1) - exchanged > threshold) & (same.max(axis=1) > threshold)
        for i in np.flatnonzero(checked)[flags]:
            transitions.append(dict(previous_frame_index=int(indices[i]), frame_index=int(indices[i + 1]),
                                    source_time_ms=float(clock[i + 1])))
    affected = [e for e in events if any(abs(e["source_time_ms"] - f["source_time_ms"]) <= NEAR_CONTACT_MS
                                        for f in transitions)]
    n = int(checked.sum())
    state = ("unavailable" if not n else "switch_like_movement_detected" if transitions
             else "no_signal_in_checked_transitions")
    return dict(status=state, checked_transitions=n, possible_transitions=len(checked),
                transitions=transitions, contacts_near_transitions=affected,
                leg_scale_px=scale, switch_scale_fraction=SWITCH_FRACTION, contact_radius_ms=NEAR_CONTACT_MS)


def item(identity, label, status, measurement_status, sentence, evidence, values, reasons, sources):
    return dict(id=identity, label=label, status=status, measurement_status=measurement_status,
                sentence=sentence, evidence=evidence, display_values=values,
                reasons=list(dict.fromkeys(reason for reason in reasons if reason)), source_ids=sources)


def value(label, number, unit, count):
    return dict(label=label, value=number, unit=unit, valid_count=count)


def gait_items(segment, frames, image):
    gait = segment["gait"]
    metrics = gait["metrics"]
    cadence, sides, difference = (metrics[key] for key in ("cadence", "step_time", "left_minus_right_step_time"))
    events = gait["events"]
    steps = gait["steps"]
    tracking = foot_tracking(frames, image, events)
    rejects = []
    for first, second in zip(events, events[1:]):
        _, rejected = calculate_steps((first, second))
        if rejected:
            rejects.append(dict(first=first, second=second, reason=next(iter(rejected))))
    evidence = dict(event_kind=gait["event_kind"], contacts=events, valid_intervals=steps,
                    rejected_pairs=rejects, rejected_pair_counts=gait["rejected_pairs"], tracking=tracking,
                    interval_count=len(steps), average_interval_s=mean(s["seconds"] for s in steps) if steps else None,
                    cadence=cadence)
    risky_interval = any(s["start_ms"] <= t["source_time_ms"] <= s["stop_ms"]
                         for s in steps for t in tracking["transitions"])
    duplicate = any(p["reason"] in ("same_side", "coincident") for p in rejects)
    quality_reasons = list(gait["quality_reasons"])
    if tracking["contacts_near_transitions"] or risky_interval:
        quality_reasons.append("foot_tracking_unstable")
    if duplicate:
        quality_reasons.append("ambiguous_contact_sequence")
    if tracking["checked_transitions"] < tracking["possible_transitions"] or tracking["status"] == "unavailable":
        quality_reasons.append("foot_tracking_unassessable")
    unreliable = bool(quality_reasons)
    if "foot_tracking_unstable" in quality_reasons:
        rhythm_sentence = "이번 구간에서는 발 위치 추적이 흔들려 걸음의 박자를 안정적으로 확인하기 어려웠습니다."
        side_sentence = "발 위치 추적이 흔들려 왼발과 오른발로 내딛는 시간을 안정적으로 확인하기 어려웠습니다."
    elif duplicate:
        rhythm_sentence = "발이 닿은 시점을 구분하기 어려워 이번 구간의 걸음 박자 판단을 보류했습니다."
        side_sentence = "발이 닿은 시점을 구분하기 어려워 왼발과 오른발로 내딛는 시간 판단을 보류했습니다."
    else:
        rhythm_sentence = "이번 구간에서는 걸음의 박자를 설명할 만큼 연속된 발 접촉을 확인하기 어려웠습니다."
        side_sentence = "이번 구간에서는 발을 내딛는 시간을 설명할 만큼 왼발과 오른발의 접촉을 확인하기 어려웠습니다."

    rhythm_values = []
    if not unreliable and cadence["value"] is not None:
        average = evidence["average_interval_s"]
        n = evidence["interval_count"]
        rhythm_sentence = f"이번 구간에서는 약 {average:.2f}초마다 한 걸음을 내디뎠습니다. 연속된 걸음 간격 {n}개로 계산하면 분당 약 {cadence['value']:.0f}걸음입니다."
        rhythm_values = [value("평균 걸음 간격", average, "s", n),
                         value("분당 걸음 수", cadence["value"], "steps/min", n)]
    rhythm = item("rhythm", "걸음 박자", "observed" if rhythm_values else "withheld",
                  "available" if cadence["value"] is not None else "unavailable", rhythm_sentence, evidence,
                  rhythm_values, [*quality_reasons, cadence["unavailable_reason"]],
                  [gait["method_id"], "first-visit-gait-results/rhythm"])

    available_sides = [side for side in SIDES if sides[side]["value"] is not None]
    side_values = []
    side_reasons = [*quality_reasons, difference["unavailable_reason"]]
    if not unreliable and available_sides:
        side_values = [value(f"{SIDE_LABELS[side]} 발로 내딛는 시간", sides[side]["value"], "s",
                             sides[side]["valid_count"]) for side in available_sides]
        side_sentence = ("이번 구간에서 확인한 왼발과 오른발의 걸음 시간을 상세에서 볼 수 있습니다."
                         if len(side_values) == 2 else
                         f"이번 구간에서는 {SIDE_LABELS[available_sides[0]]} 발로 내딛는 시간을 확인했습니다.")
        side_reasons.append("side_direction_requires_reproduction")
    side_state = "available" if len(available_sides) == 2 else "partial" if available_sides else "unavailable"
    side_result = item("step_time", "좌우 걸음 시간", "limited" if side_values else "withheld", side_state,
                       side_sentence, dict(step_time=sides, left_minus_right=difference,
                                           sign_convention="left_minus_right", tracking=tracking),
                       side_values, side_reasons, [gait["method_id"], "first-visit-gait-results/step-time"])
    return [rhythm, side_result]


def knee_item(segment, frames):
    knees = segment["knee"]
    supported = knees["views"]["supported"]
    times = [f["source_time_ms"] for f in frames]
    indices = [f["index"] for f in frames]
    n = len(times)
    support = {side: [angle is not None for angle in supported["curve_deg"][side]] for side in SIDES}
    common = [left and right for left, right in zip(support["left"], support["right"])]
    dt = np.diff(times)
    contiguous = n >= 2 and all(gap == 1 for gap in np.diff(indices)) and bool(np.all(dt <= 2 * np.median(dt)))

    def coverage(flags):
        seen = [t for t, ok in zip(times, flags) if ok]
        return dict(valid_frames=len(seen), fraction=len(seen) / n if n else None,
                    first_source_ms=seen[0] if seen else None, last_source_ms=seen[-1] if seen else None,
                    span_ms=seen[-1] - seen[0] if len(seen) >= 2 else 0)

    side_coverage = {side: coverage(support[side]) for side in SIDES}
    common_coverage = coverage(common)
    ranges = supported["bilateral"]
    measured = [side for side in SIDES if supported[side]["range_deg"] is not None]
    complete_sides = [side for side in measured if contiguous and side_coverage[side]["valid_frames"] == n]
    bilateral = len(complete_sides) == 2
    values, reasons = [], []
    if bilateral:
        sentence = "이번 구간에서 양쪽 무릎이 움직인 범위를 확인했습니다. 각각의 움직임 폭은 상세에서 볼 수 있습니다."
        values = [value(f"{SIDE_LABELS[side]} 무릎 움직임 폭", ranges[side + "_range_deg"], "deg",
                        common_coverage["valid_frames"]) for side in SIDES]
    else:
        reasons.append("insufficient_bilateral_knee_coverage")
        if len(complete_sides) == 1:
            side = complete_sides[0]
            other = next(s for s in SIDES if s != side)
            sentence = (f"{SIDE_LABELS[other]} 무릎을 관찰할 수 있었던 구간이 짧아, 양쪽 무릎의 움직임 차이는 판단하기 어렵습니다.")
            values = [value(f"{SIDE_LABELS[side]} 무릎 움직임 폭", supported[side]["range_deg"], "deg",
                            side_coverage[side]["valid_frames"])]
        elif measured:
            sentence = "무릎이 잘 보인 부분이 고르지 않아, 이번 구간 전체의 무릎 움직임 차이는 판단하기 어렵습니다."
        else:
            sentence = "이번 구간에서는 무릎이 충분히 보이지 않아 움직임 폭을 확인하기 어려웠습니다."
    evidence = dict(source_frame_count=n, source_frames_contiguous=contiguous, coverage=side_coverage,
                    simultaneous_coverage=common_coverage, simultaneous_ranges_deg=ranges,
                    supported_range_deg={side: supported[side]["range_deg"] for side in SIDES},
                    method="2D angle: 95th minus 5th percentile")
    state = "available" if len(measured) == 2 else "partial" if measured else "unavailable"
    return item("knee", "무릎 움직임", "observed" if bilateral else "limited" if values else "withheld",
                state, sentence, evidence, values, reasons, [knees["method_id"], "first-visit-gait-results/knee"])


def gdi_item(gdi):
    models = {}
    values = []
    for name, label in (("baseline", "기존 모델"), ("normal100", "정상100 모델")):
        row = gdi["models"][name]
        models[name] = {key: row[key] for key in ("status", "value", "unit", "unavailable_reason")}
        if row["status"] == "available":
            values.extend(value(f"{label} · {SIDE_LABELS[side]}", row["value"][side], row["unit"], None)
                          for side in SIDES if row["value"][side] is not None)
    state = "available" if len(values) == 4 else "partial" if values else "unavailable"
    sentence = ("보행 패턴의 모델 추정값은 연구용 상세에서 볼 수 있습니다." if values else
                "이 구간의 보행 패턴 추정값은 연구용 상세에서도 계산값이 없습니다.")
    return item("gdi", "보행 패턴 추정값 (GDI)", "research_only", state, sentence,
                dict(models=models, input=gdi.get("input"), unavailable_reason=gdi.get("unavailable_reason")), values,
                [gdi.get("unavailable_reason"), *(m["unavailable_reason"] for m in models.values())],
                ["gdi-segment/1", "first-visit-gait-results/gdi"])


def build_report(source_id, segments, frame_groups, image, detection):
    rows = []
    for s, frames in zip(segments, frame_groups, strict=True):
        rows.append(dict(
            **{key: s[key] for key in ("segment_id", "direction", "start_ms", "stop_ms")},
            metrics=[*gait_items(s, frames, image), knee_item(s, frames), gdi_item(s["gdi"])]))
    notice = None if rows else "이번 영상에서 걸음을 설명할 수 있는 보행 구간을 찾기 어려웠습니다."
    return dict(schema=SCHEMA, policy_version=POLICY, language="ko-KR", scope="video",
                source_id=source_id, notice=notice, detection_status=detection["status"], segments=rows)
