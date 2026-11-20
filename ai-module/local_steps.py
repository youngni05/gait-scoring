"""Recover heel-contact intervals within uninterrupted parts of a rejected pass."""
from math import isfinite
from statistics import median

from core import mapped_pose
from vendor import gait_events


def local_step_intervals(pose, analysis):
    mapped = mapped_pose(pose)
    frames = {f['index']: f for f in mapped['frames']}
    reports = {s['segment_id']: s for s in analysis['current_report']['segments']}
    result = {}
    for segment in analysis['segments']:
        gait = segment['gait']
        if segment['direction'] not in ('image_left', 'image_right'):
            continue
        rhythm = next(m for m in reports[segment['segment_id']]['metrics'] if m['id'] == 'rhythm')
        tracking = rhythm['evidence']['tracking']
        # Existing events retain their original timestamps when the full pass
        # was measurable. The app removes only intervals near tracking switches.
        if not gait['quality_reasons'] and tracking['checked_transitions'] == tracking['possible_transitions']:
            continue
        selected = [frames[i] for i in segment['observation']['source_indices']]
        if len(selected) < 7:
            continue
        image = mapped['image']
        max_gap = 2 * median(b['source_time_ms'] - a['source_time_ms'] for a, b in zip(selected, selected[1:]))
        changes = {t['frame_index'] for t in tracking['transitions']}
        runs, run = [], []
        for frame in selected:
            joints = frame['joints']
            visible = joints is not None and all(
                x is not None and y is not None and v is not None and isfinite(v)
                and 0 <= x <= image['width'] and 0 <= y <= image['height']
                for x, y, v, _ in (joints[i] for i in gait_events.JOINTS))
            if run and (not visible or frame['index'] in changes or frame['index'] != run[-1]['index'] + 1
                        or frame['source_time_ms'] - run[-1]['source_time_ms'] > max_gap):
                runs.append(run)
                run = []
            if visible:
                run.append(frame)
        if run:
            runs.append(run)
        steps = []
        for run in runs:
            if len(run) < 7:
                continue
            measured = gait_events.measure_payload(dict(mapped, frames=run), segment['start_ms'], segment['stop_ms'],
                                                   segment['direction'].removeprefix('image_'))
            steps.extend(measured['steps'])
        result[segment['segment_id']] = steps
    return result
