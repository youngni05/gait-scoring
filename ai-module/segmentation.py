"""Orientation and leg-motion candidates on native source times, including tracked cameras."""
from math import hypot, nextafter, inf
from collections import Counter
import numpy as np

METHOD = "sidewalk-orientation-native/1"
SETTINGS = dict(visibility=0.5, foot_horizontal_fraction=0.5, min_cue_leg_fraction=0.02,
                vote_window_frames=5, minimum_votes=3, turn_margin_ms=200,
                min_frames=7, min_span_ms=500, min_motion_leg_fraction=0.10, max_gap_factor=2)


def point(points, i, w, h, visible=False):
    p = points[i]
    if p[0] is None or p[1] is None or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1):
        return None
    if visible and (p[3] is None or p[3] < SETTINGS["visibility"]):
        return None
    return p[0] * w, p[1] * h


def cue(frame, w, h):
    points = frame["landmarks"]
    if points is None:
        return None, "pose_absent", None, None
    hips = [point(points, i, w, h) for i in (23, 24)]
    ankles = [point(points, i, w, h) for i in (27, 28)]
    if any(p is None for p in hips + ankles):
        return None, "required_coordinate_missing_or_outside_image", None, None
    scale = sum(hypot(a[0]-b[0], a[1]-b[1]) for a, b in zip(hips, ankles)) / 2
    if scale <= 1e-9:
        return None, "degenerate_leg_scale", None, None
    relative = (ankles[0][0] - ankles[1][0]) / scale
    feet = []
    for heel, toe in ((29, 31), (30, 32)):
        a, b = point(points, heel, w, h, True), point(points, toe, w, h, True)
        if a is None or b is None:
            continue
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = hypot(dx, dy)
        if length >= 0.02 * scale and abs(dx) >= 0.5 * length:
            feet.append(1 if dx > 0 else -1)
    face = None
    nose = point(points, 0, w, h, True)
    ears = [point(points, i, w, h, True) for i in (7, 8)]
    ears = [p for p in ears if p is not None]
    if nose is not None and ears:
        dx = nose[0] - sum(p[0] for p in ears) / len(ears)
        if abs(dx) >= 0.02 * scale:
            face = 1 if dx > 0 else -1
    if len(set(feet)) > 1 or feet and face is not None and feet[0] != face:
        return None, "direction_cues_disagree", relative, scale
    if len(feet) == 2 or len(feet) == 1 and face == feet[0]:
        return feet[0], None, relative, scale
    return None, "heading_uncertain", relative, scale


def detect(pose, max_segments=16):
    frames = pose["frames"]
    w, h = pose["image"]["width"], pose["image"]["height"]
    if not frames:
        return dict(mode="automatic", method_id=METHOD, status="unavailable", segments=[],
                    excluded=[], settings=SETTINGS, boundaries=[], limits=dict(max_segments=max_segments),
                    assumptions=["forward_side_view_walking", "facing_cue_used_for_travel_direction"],
                    unsupported=["backward_walking_direction", "physical_path_straightness_without_reference"],
                    reasons=["no_source_frames"])
    times = np.array([f["source_time_ms"] for f in frames], float)
    indices = [f["index"] for f in frames]
    median_gap = float(np.median(np.diff(times))) if len(times) > 1 else None
    cues = [cue(f, w, h) for f in frames]
    blocks = []
    boundary_rows = []
    start = 0
    for i in range(1, len(frames)):
        if indices[i] != indices[i-1] + 1 or times[i]-times[i-1] > SETTINGS["max_gap_factor"] * median_gap:
            blocks.append((start, i))
            boundary_rows.append(dict(after_frame_index=indices[i-1], before_frame_index=indices[i], reason="source_frame_gap",
                                      gap_ms=float(times[i]-times[i-1])))
            start = i
    blocks.append((start, len(frames)))
    labels = [None] * len(frames)
    reasons = [c[1] or "insufficient_heading_votes" for c in cues]
    for begin, end in blocks:
        for i in range(begin, end):
            if cues[i][1] is not None:
                continue
            votes = Counter(cues[j][0] for j in range(max(begin, i-2), min(end, i+3)) if cues[j][0] is not None)
            if votes:
                direction, count = votes.most_common(1)[0]
                if count >= SETTINGS["minimum_votes"] and count > votes.get(-direction, 0):
                    labels[i], reasons[i] = direction, None
        valid = [i for i in range(begin, end) if labels[i] is not None]
        turns = [i for a, i in zip(valid, valid[1:]) if labels[a] != labels[i]]
        for pivot in turns:
            for i in range(begin, end):
                if abs(times[i]-times[pivot]) <= SETTINGS["turn_margin_ms"]:
                    labels[i], reasons[i] = None, "direction_transition_margin"
    segments = []
    # Build runs without bridging gaps, unknown frames or direction changes.
    for begin, end in blocks:
        i = begin
        while i < end:
            if labels[i] is None:
                i += 1
                continue
            stop = i + 1
            while stop < end and labels[stop] == labels[i]:
                stop += 1
            size, span = stop-i, float(times[stop-1]-times[i])
            movement = [cues[j][2] for j in range(i, stop) if cues[j][2] is not None]
            motion = float(np.quantile(movement, .95)-np.quantile(movement, .05)) if len(movement) >= 2 else 0
            reason = None
            if size < SETTINGS["min_frames"] or span < SETTINGS["min_span_ms"]:
                reason = "insufficient_segment_extent"
            elif motion < SETTINGS["min_motion_leg_fraction"]:
                reason = "insufficient_gait_motion"
            elif len(segments) >= max_segments:
                reason = "segment_service_limit"
            if reason:
                for j in range(i, stop):
                    labels[j], reasons[j] = None, reason
            else:
                segments.append(dict(segment_id=f"auto-{len(segments)+1}", start_ms=float(times[i]),
                    stop_ms=nextafter(float(times[stop-1]), inf),
                    direction="image_right" if labels[i] == 1 else "image_left",
                    first_frame_index=indices[i], last_frame_index=indices[stop-1], frame_count=size,
                    observed_span_ms=span, motion_leg_fraction=motion, status="sidewalk_candidate"))
            i = stop
    excluded = []
    i = 0
    while i < len(frames):
        if reasons[i] is None:
            i += 1
            continue
        stop = i + 1
        while stop < len(frames) and reasons[stop] == reasons[i] and indices[stop] == indices[stop-1] + 1:
            stop += 1
        excluded.append(dict(first_frame_index=indices[i], last_frame_index=indices[stop-1],
            start_ms=float(times[i]), stop_ms=nextafter(float(times[stop-1]), inf), frames=stop-i, reason=reasons[i]))
        i = stop
    return dict(mode="automatic", method_id=METHOD, status="candidates_found" if segments else "unavailable",
                segments=segments, excluded=excluded, boundaries=boundary_rows, settings=SETTINGS,
                assumptions=["forward_side_view_walking", "facing_cue_used_for_travel_direction"],
                unsupported=["backward_walking_direction", "physical_path_straightness_without_reference"],
                limits=dict(max_segments=max_segments), reasons=[] if segments else ["no_eligible_sidewalk_candidate"])
