"""Knee motion from time-aligned, upright MediaPipe image landmarks."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

JOINTS = {'left': (23, 25, 27), 'right': (24, 26, 28)}
VIEWS = ('observed', 'supported')
POSE_INTERFACE = Path(__file__).resolve().parent.with_name('20261117-pose-interface')


def finite_curve(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def describe(values):
    usable = values[np.isfinite(values)]
    endpoints = np.quantile(usable, [.05, .95]) if usable.size >= 2 else None
    return dict(valid_frames=int(usable.size), median_deg=float(np.median(usable)) if usable.size else None,
                p05_deg=float(endpoints[0]) if endpoints is not None else None,
                p95_deg=float(endpoints[1]) if endpoints is not None else None,
                range_deg=float(np.diff(endpoints)[0]) if endpoints is not None else None,
                range_unavailable_reason=None if endpoints is not None else 'fewer_than_two_valid_frames')


def calculate(landmarks, source_ms, width, height, units='normalized'):
    """Return angle arrays and quality flags. Anatomical labels follow MediaPipe."""
    raw = np.asarray(landmarks, dtype=float)
    time = np.asarray(source_ms, dtype=float)
    if time.ndim != 1 or raw.shape != (len(time), 33, 5):
        raise ValueError('Expected N timestamp rows and N x 33 x 5 MediaPipe landmarks')
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Source timestamps must be finite and strictly increasing')
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError('Image dimensions must be positive')
    if units not in ('normalized', 'pixel'):
        raise ValueError('Coordinates must be upright image-normalized or pixel x/y')

    xy = raw[:, :, :2] * ([width, height] if units == 'normalized' else [1, 1])
    angles = {key: {} for key in VIEWS}
    quality = {}
    for side, ids in JOINTS.items():
        joints = xy[:, ids]
        finite = np.isfinite(joints).all(axis=(1, 2))
        with np.errstate(invalid='ignore', over='ignore'):
            a = joints[:, 0] - joints[:, 1]
            b = joints[:, 2] - joints[:, 1]
            valid = finite & (np.hypot(a[:, 0], a[:, 1]) * np.hypot(b[:, 0], b[:, 1]) > 1e-10)
            det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            dot = np.sum(a * b, axis=1)
            curve = 180 - np.degrees(np.arctan2(np.abs(det), dot))
        inside = ((0 <= joints) & (joints <= [width, height])).all(axis=(1, 2))
        visibility = (raw[:, ids, 3] >= .5).all(axis=1)
        presence = (raw[:, ids, 4] >= .5).all(axis=1)
        angles['observed'][side] = np.where(valid, curve, np.nan)
        angles['supported'][side] = np.where(valid & inside & visibility, curve, np.nan)
        quality[side] = dict(nonfinite_xy_frames=int((~finite).sum()), degenerate_frames=int((finite & ~valid).sum()),
                             observed_outside_image_frames=int((valid & ~inside).sum()),
                             observed_low_or_missing_visibility_frames=int((valid & ~visibility).sum()),
                             supported_low_or_missing_presence_frames=int((valid & inside & visibility & ~presence).sum()))
    return angles, quality


def summarize(angles):
    left, right = angles['left'], angles['right']
    common = np.isfinite(left) & np.isfinite(right)
    l_joint, r_joint = describe(left[common]), describe(right[common])
    difference = left[common] - right[common]
    gap = l_joint['range_deg'] - r_joint['range_deg'] if common.sum() >= 2 else None
    return dict(left=describe(left), right=describe(right), bilateral=dict(
        common_valid_frames=int(common.sum()), left_range_deg=l_joint['range_deg'], right_range_deg=r_joint['range_deg'],
        left_minus_right_range_deg=gap, absolute_range_gap_deg=abs(gap) if gap is not None else None,
        median_left_minus_right_angle_deg=float(np.median(difference)) if difference.size else None))


def measure(landmarks, source_ms, width, height, units='normalized'):
    angles, quality = calculate(landmarks, source_ms, width, height, units)
    return dict(source_ms=np.asarray(source_ms, dtype=float).tolist(), frames=len(source_ms), geometry='upright_image_2d',
                angle_units='degrees', sides='anatomical', quality=quality, views={view: {
                    'curve_deg': {side: finite_curve(angles[view][side]) for side in JOINTS},
                    'simultaneous_left_minus_right_deg': finite_curve(angles[view]['left'] - angles[view]['right']),
                    **summarize(angles[view]),
                } for view in VIEWS})


def measure_packet(packet, start_ms=None, stop_ms=None):
    spec = importlib.util.spec_from_file_location('shared_pose_adapter', POSE_INTERFACE / 'adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapted, _ = adapter.adapt(packet)
    frames = adapted['frames']
    if not frames and (start_ms is None or stop_ms is None):
        raise ValueError('An empty capture requires explicit segment bounds')
    start_ms = frames[0]['source_time_ms'] if start_ms is None else start_ms
    stop_ms = float(np.nextafter(frames[-1]['source_time_ms'], np.inf)) if stop_ms is None else stop_ms
    if not np.isfinite([start_ms, stop_ms]).all() or start_ms >= stop_ms:
        raise ValueError('Expected finite segment bounds with start before stop')
    selected = [f for f in frames if start_ms <= f['source_time_ms'] < stop_ms]
    time = [f['source_time_ms'] for f in selected]
    points = np.full((len(time), 33, 5), np.nan)
    ids = adapter.CONTRACT['output']['mediapipe_indices']
    for i, f in enumerate(selected):
        if f['joints'] is not None:
            joints = np.asarray(f['joints'], dtype=float)
            points[i, ids, :2] = joints[:, :2]
            points[i, ids, 3:] = joints[:, 2:]
    image = adapted['image']
    motion = measure(points, time, image['width'], image['height'], units='pixel')
    metrics = {}
    for view, result in motion['views'].items():
        def scalar(value, n):
            return dict(value=value, unit='deg', valid_count=n,
                        unavailable_reason='fewer_than_two_valid_frames' if value is None else None)
        metrics[view + '_knee_range'] = {side: scalar(result[side]['range_deg'], result[side]['valid_frames']) for side in JOINTS}
        bilateral = result['bilateral']
        metrics[view + '_left_minus_right_range'] = scalar(bilateral['left_minus_right_range_deg'], bilateral['common_valid_frames'])
    return dict(schema='measurement-metrics/1', method_id='knee-flexion-2d/1', source_id=packet['source_id'],
                start_ms=start_ms, stop_ms=stop_ms, extractor=adapted['extractor'], metrics=metrics,
                selected_frame_indices=[f['index'] for f in selected],
                pose_unavailable_reasons=[f['why_unavailable'] for f in selected], knee_motion=motion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose', type=Path, help='measurement-pose/1 JSON or archived Test02 NPZ')
    parser.add_argument('output_json', type=Path)
    parser.add_argument('--start', type=int, default=0, help='Inclusive frame index')
    parser.add_argument('--stop', type=int, help='Exclusive frame index')
    parser.add_argument('--start-ms', type=float, help='Inclusive source timestamp for JSON')
    parser.add_argument('--stop-ms', type=float, help='Exclusive source timestamp for JSON')
    args = parser.parse_args()
    if args.pose.suffix.lower() == '.json':
        if args.start or args.stop is not None:
            parser.error('Select JSON segments with source timestamps')
        result = measure_packet(json.loads(args.pose.read_text(encoding='utf-8-sig')), args.start_ms, args.stop_ms)
    else:
        if args.start_ms is not None or args.stop_ms is not None:
            parser.error('Select NPZ segments with frame indices')
        with np.load(args.pose) as pose:
            stop = len(pose['source_ms']) if args.stop is None else args.stop
            if not 0 <= args.start <= stop <= len(pose['source_ms']):
                parser.error('Frame bounds lie outside the stored pose')
            selected = slice(args.start, stop)
            result = measure(pose['landmarks'][selected], pose['source_ms'][selected], int(pose['width']), int(pose['height']))
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
