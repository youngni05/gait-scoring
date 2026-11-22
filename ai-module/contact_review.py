"""Apply the timing-only hold from abc_detector to a MediaPipe heel-event stream."""
from itertools import groupby
from statistics import mean

from vendor.gait_events import calculate_steps, measure_payload, metric


METHOD = "relative-heel-native-pts/repeated-side-gap-1"


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
