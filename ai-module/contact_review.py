"""Apply the timing-only hold from abc_detector to a MediaPipe heel-event stream."""
from itertools import groupby
from statistics import mean

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from vendor.gait_events import calculate_steps, measure_payload, metric


METHOD = "relative-heel-native-pts/repeated-side-gap-1"
SEPARATION_METHOD = "bilateral-foot-separation/native-pts-1"


def recover_rhythm(gait, frames, image, tracking):
    """Recover timing when the forward foot repeatedly receives the same anatomical label."""
    if (gait["quality_reasons"] or len(tracking["transitions"]) < 2 or
            tracking["checked_transitions"] != tracking["possible_transitions"] or not gait["events"]):
        return
    switches = [t["source_time_ms"] for t in tracking["transitions"]]
    safe = [s for s in gait["steps"] if not any(s["start_ms"] - tracking["contact_radius_ms"] <= t
            <= s["stop_ms"] + tracking["contact_radius_ms"] for t in switches)]
    if any(a["stop_ms"] == b["start_ms"] and a["landing_side"] != b["landing_side"] for a, b in zip(safe, safe[1:])):
        return
    clock = np.array([f["source_time_ms"] for f in frames])
    dt = np.median(np.diff(clock))
    # Filter only densely sampled native video. Do not smooth across missing observations.
    if not 8 <= dt <= 50 or np.max(np.diff(clock)) > 1.5 * dt:
        return
    feet = np.array([f["landmarks"][27:33] for f in frames], dtype=float)
    visible = (np.isfinite(feet[:, :, [0, 1, 3]]).all(axis=(1, 2)) &
               ((feet[:, :, :2] >= 0) & (feet[:, :, :2] <= 1)).all(axis=(1, 2)) &
               (feet[:, :, 3] >= .5).all(axis=1))
    if not visible.any():
        return
    lo, hi = np.flatnonzero(visible)[[0, -1]]
    if not visible[lo:hi+1].all() or hi - lo < 6:
        return
    signed = (feet[lo:hi+1, ::2, 0] - feet[lo:hi+1, 1::2, 0]) * image["width"]
    separation = np.median(abs(signed), axis=1)
    # Median removes a one-frame collapse. The smoothing widths are in source time.
    radius = max(1, round(33 / dt))
    width = gaussian_filter1d(median_filter(separation, size=2 * radius + 1), sigma=50 / dt)
    prominence = max(.2 * np.quantile(width, .9), .08 * tracking["leg_scale_px"])
    peaks, _ = find_peaks(width, prominence=prominence)
    if len(peaks) < 5:
        return
    signs = np.sign(signed[peaks])
    if not ((signs > 0).all() or (signs < 0).all()):
        return
    indices = lo + peaks
    times = clock[indices]
    gaps = np.diff(times) / 1000
    if np.any((gaps < .2) | (gaps > 1.5)):
        return
    heel_times = np.array([e["source_time_ms"] for e in gait["events"]])
    if np.mean(np.min(abs(times[:, None] - heel_times), axis=1) <= 150) < .8:
        return
    gait["heel_evidence"] = {k: gait[k] for k in ("method_id", "events", "steps", "rejected_pairs", "metrics")}
    events = [dict(frame_index=frames[i]["index"], source_time_ms=float(clock[i]), side="unknown") for i in indices]
    steps = [dict(start_ms=a["source_time_ms"], stop_ms=b["source_time_ms"], seconds=float(s), landing_side="unknown")
             for a, b, s in zip(events, events[1:], gaps)]
    count = len(steps)
    gait.update(method_id=SEPARATION_METHOD, events=events, steps=steps, rejected_pairs={}, metrics=dict(
        cadence=metric(float(60 / gaps.mean()), "steps/min", count), valid_steps=metric(count, "count", count),
        step_time={s: metric(None, "s", 0, "unresolved_landing_side") for s in ("left", "right")},
        left_minus_right_step_time=metric(None, "s", 0, "unresolved_landing_side")))


def held_indices(events):
    """Only a pair with exactly one unusually close opposite-side neighbor qualifies."""
    held = set()
    for _, group in groupby(range(len(events)), key=lambda i: events[i]["side"]):
        indices = list(group)
        if len(indices) != 2:
            continue
        first, last = indices
        span = events[last]["source_time_ms"] - events[first]["source_time_ms"]
        flagged = []
        for i, neighbor in ((first, first - 1), (last, last + 1)):
            if 0 <= neighbor < len(events):
                gap = abs(events[i]["source_time_ms"] - events[neighbor]["source_time_ms"])
                if gap < span / 2:
                    flagged.append(i)
        if len(flagged) == 1:
            held.add(flagged[0])
    return held


def measure_contacts(payload, start, stop, direction, tracking):
    gait = measure_payload(payload, start, stop, direction)
    events = gait["events"]
    held = set()
    if tracking["possible_transitions"] and tracking["checked_transitions"] == tracking["possible_transitions"]:
        switches = [t["source_time_ms"] for t in tracking["transitions"]]
        held = {i for i in held_indices(events)
                if all(abs(events[i]["source_time_ms"] - t) > tracking["contact_radius_ms"] for t in switches)}
    gait.update(method_id=METHOD, held_contacts=[e for i, e in enumerate(events) if i in held])
    if not held:
        return gait
    kept = [e for i, e in enumerate(events) if i not in held]
    steps, rejects = calculate_steps(kept)
    durations = [s["seconds"] for s in steps]
    side_time = {}
    for side in ("left", "right"):
        times = [s["seconds"] for s in steps if s["landing_side"] == side]
        side_time[side] = metric(mean(times) if len(times) >= 2 else None, "s", len(times), "insufficient_side_steps")
    left, right = (side_time[side]["value"] for side in ("left", "right"))
    count = len(steps)
    gait.update(events=kept, steps=steps, rejected_pairs=rejects, metrics=dict(
        cadence=metric(60 / mean(durations) if count >= 2 else None, "steps/min", count, "insufficient_valid_steps"),
        valid_steps=metric(count, "count", count),
        step_time=side_time,
        left_minus_right_step_time=metric(left - right if left is not None and right is not None else None,
                                          "s", count, "insufficient_steps_each_side"),
    ))
    return gait
