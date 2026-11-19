"""Estimate step intervals from upright MediaPipe joints and original frame clocks."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.signal import find_peaks

from .io_support import read, save

JOINTS = [0, 5, 3, 8]  # left hip, right hip, left heel, right heel in measurement-joints/1
JOINT_NAMES = ['left_hip', 'right_hip', 'left_heel', 'right_heel']
METHOD = 'relative-heel-native-pts/1'


def smooth(t, values):
    filtered = np.empty_like(values)
    for i in range(len(t)):
        start = min(max(i - 3, 0), len(t) - 7)
        ids = slice(start, start + 7)
        clocks = (t[ids] - t[i]) / (t[start + 6] - t[start])
        design = np.column_stack((np.ones(7), clocks, clocks**2))
        filtered[i] = np.linalg.lstsq(design, values[ids], rcond=None)[0][0]
    return filtered


def calculate_steps(events):
    rejected = Counter()
    steps = []
    for previous, current in zip(events, events[1:]):
        dt = (current['source_time_ms'] - previous['source_time_ms']) / 1000
        if dt <= 0:
            rejected['coincident'] += 1
        elif current['side'] == previous['side']:
            rejected['same_side'] += 1
        elif not .2 <= dt <= 1.5:
            rejected['out_of_bounds'] += 1
        else:
            steps.append(dict(landing_side=current['side'], seconds=dt,
                              start_ms=previous['source_time_ms'], stop_ms=current['source_time_ms']))
    return steps, dict(rejected)


def metric(value, unit, count, reason=None):
    return dict(value=value, unit=unit, valid_count=count, unavailable_reason=reason if value is None else None)


def measure(source_id, start_ms, stop_ms, direction, width, height, ids, clocks, joints, pose_available):
    if (direction not in ('left','right') or not np.isfinite([start_ms,stop_ms,width,height]).all()
            or not start_ms < stop_ms or width <= 0 or height <= 0):
        raise ValueError('Expected positive image dimensions, a half-open interval and screen travel direction')
    ids = np.asarray(ids, dtype=int)
    clocks = np.asarray(clocks, dtype=float)
    poses = np.asarray(joints, dtype=float).reshape(-1, 4, 4)
    available = np.asarray(pose_available, dtype=bool)
    if len({len(ids), len(clocks), len(poses), len(available)}) != 1:
        raise ValueError('Frame arrays must agree')
    if not np.isfinite(clocks).all() or np.any(np.diff(clocks) <= 0) or np.any(np.diff(ids) <= 0):
        raise ValueError('Source frame indices and clocks must strictly increase')
    mask = (clocks >= start_ms) & (clocks < stop_ms)
    ids, clocks, poses, available = ids[mask], clocks[mask], poses[mask], available[mask]
    n = len(clocks)
    xy, visible = poses[..., :2], poses[..., 2]
    reasons = []
    if n < 7:
        reasons.append('fewer_than_7_source_frames')
    if n and not available.all():
        reasons.append('pose_absent')
    if n and not np.isfinite(poses[..., :3]).all():
        reasons.append('missing_required_xy_or_visibility')
    out_of_frame = np.isfinite(xy) & ((xy < 0) | (xy > [width, height]))
    if out_of_frame.any():
        reasons.append('required_landmark_outside_image')
    if n and np.isfinite(visible).all() and np.any(np.median(visible, axis=0) < .5):
        reasons.append('required_median_visibility_below_0.5')
    intervals = np.diff(clocks)
    if (n > 1 and np.any(np.diff(ids) != 1)) or (n > 2 and np.any(intervals > 2 * np.median(intervals))):
        reasons.append('source_frame_gap')
    events = []
    if not reasons:
        sign = -1 if direction == 'left' else 1
        relative = sign * (xy[:, 2:, 0] - xy[:, :2, 0].mean(1)[:,None])
        signal = smooth(clocks, relative)
        for column, side in enumerate(('left','right')):
            track = signal[:, column]
            peaks, _ = find_peaks(track, prominence=.15 * np.ptp(track))
            kept = []
            for peak in sorted(peaks, key=lambda p: (-track[p], p)):
                if all(abs(clocks[peak] - clocks[other]) >= 400 for other in kept):
                    kept.append(peak)
            events.extend(dict(side=side, frame_index=int(ids[i]), source_time_ms=float(clocks[i])) for i in kept)
    events.sort(key=lambda e: (e['source_time_ms'],e['side']))
    steps, rejects = calculate_steps(events)
    by_side = {side:[step['seconds'] for step in steps if step['landing_side']==side] for side in ('left','right')}
    cause = reasons[0] if reasons else None
    all_times = [step['seconds'] for step in steps]
    cadence = 60 / mean(all_times) if len(all_times) >= 2 else None
    side_mean = {side:mean(values) if len(values)>=2 else None for side,values in by_side.items()}
    asymmetry = side_mean['left'] - side_mean['right'] if all(v is not None for v in side_mean.values()) else None
    return dict(schema='measurement-metrics/1',source_id=source_id,start_ms=start_ms,stop_ms=stop_ms,
                method_id=METHOD, event_kind='pose_estimate', source_frames=n, quality_reasons=reasons,
                events=events, steps=steps, rejected_pairs=rejects, metrics={
                    'cadence':metric(cadence,'steps/min',len(steps),cause or 'insufficient_valid_steps'),
                    'valid_steps':metric(None if cause else len(steps),'count',len(steps),cause),
                    'step_time':{side:metric(side_mean[side],'s',len(by_side[side]),cause or 'insufficient_side_steps')
                                 for side in by_side},
                    'left_minus_right_step_time':metric(asymmetry,'s',len(steps),cause or 'insufficient_steps_each_side'),
                })


def measure_payload(payload, start_ms, stop_ms, direction):
    if payload['schema'] != 'measurement-joints/1' or payload['image']['orientation'] != 'upright_unmirrored':
        raise ValueError('Expected measurement-joints/1 in upright unmirrored geometry')
    if payload['clock']['unit'] != 'ms' or [payload['joint_order'][i] for i in JOINTS] != JOINT_NAMES:
        raise ValueError('Expected millisecond source clock and canonical anatomical joint order')
    frames = payload['frames']
    image = payload['image']
    return measure(payload['source_id'], start_ms, stop_ms, direction, image['width'], image['height'],
                   [f['index'] for f in frames], [f['source_time_ms'] for f in frames],
                   [[f['joints'][i] for i in JOINTS] if f['joints'] is not None else [[None]*4]*4 for f in frames],
                   [f['joints'] is not None for f in frames])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start-ms', type=float, required=True)
    parser.add_argument('--stop-ms', type=float, required=True)
    parser.add_argument('--direction', choices=('left','right'), required=True)
    args = parser.parse_args()
    save(args.output, measure_payload(read(args.input), args.start_ms, args.stop_ms, args.direction))
