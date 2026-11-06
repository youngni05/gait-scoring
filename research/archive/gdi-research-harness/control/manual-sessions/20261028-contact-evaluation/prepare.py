"""Build a local, unlabelled reference-review packet from existing research videos."""
import sys, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '20261028-camera-step-pilot'
sys.path.insert(0, str(SOURCE / 'deps'))
import cv2

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_manifest = json.loads((SOURCE / 'video_manifest.json').read_text())
original = {entry['camera']: entry for entry in source_manifest}
predictions_path = SOURCE / 'short_clip_results.json'
predictions_hash = digest(predictions_path)
predictions = json.loads(predictions_path.read_text())
start, end, fps = 50, 300, 50
records = []
images = []
for camera in ['03', '06']:
    path = SOURCE / f'camera{camera}.mp4'
    assert digest(path) == original[camera]['sha256']
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == fps
    folder = ROOT / 'frames' / camera
    folder.mkdir(parents=True, exist_ok=True)
    timestamps = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamps.append(timestamp)
        assert frame.shape[:2] == (600, 800)
        assert abs(timestamp - i * 20) < 0.1
        if start <= i <= end:
            target = folder / f'{i:04d}.png'
            assert cv2.imwrite(str(target), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            # The displayed image must preserve the decoded original pixels.
            assert np.array_equal(cv2.imread(str(target)), frame)
            images.append({'camera': camera, 'frame': i, 'time_s': i / fps,
                           'file': str(target.relative_to(ROOT)).replace('\\', '/'),
                           'sha256': digest(target)})
        i += 1
    cap.release()
    assert i == 511 and all(b > a for a, b in zip(timestamps, timestamps[1:]))
    assert digest(path) == original[camera]['sha256']
    records.append({'camera': camera, 'source_sha256': original[camera]['sha256'],
                    'frames': i, 'fps': fps, 'width': 800, 'height': 600,
                    'timestamp_max_error_ms': max(abs(t - j * 20) for j, t in enumerate(timestamps))})

cases = []
for r in predictions:
    assert r['frames'] == round((r['end_s'] - r['start_s']) * fps)
    case = dict(r)
    case.pop('elapsed_s')
    case['start_frame_inclusive'] = round(r['start_s'] * fps)
    case['end_frame_exclusive'] = round(r['end_s'] * fps)
    case['candidate_frames'] = [round(t * fps) for t in r['peak_times_s']]
    assert all(case['start_frame_inclusive'] <= f < case['end_frame_exclusive'] for f in case['candidate_frames'])
    cases.append(case)

packet = {
    'packet_id': 'tnt15-sg00-contact-review-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'source': 'https://www.tnt.uni-hannover.de/project/TNT15/',
    'use': 'Local scientific review only; preserve original research-use restrictions.',
    'people': 1, 'walks': 1, 'views': 2,
    'review_start_frame': start, 'review_end_frame': end, 'fps': fps,
    'analysis_windows': [{'start_frame_inclusive': 75, 'end_frame_exclusive': 175},
                         {'start_frame_inclusive': 175, 'end_frame_exclusive': 275},
                         {'start_frame_inclusive': 75, 'end_frame_exclusive': 275}],
    'reference_status': 'No provider or human-reviewed contact labels available. Blank annotation forms only.',
    'event_definition': 'Initial contact: the first visible contact of any part of one foot with the ground. Record an inclusive frame range when uncertain; do not infer through occlusion.',
    'cross_camera_timing': 'Per-video timestamps checked. No independent inter-camera offset measurement; annotate each view separately.',
    'scope': 'Preparation for a measurement experiment; no model fitting, pose re-inference, clinical threshold, or accuracy estimate.',
    'limits': ['Contains direction changes; not a straight-walking-only benchmark.',
               'Overlapping windows and two views are not independent people.',
               'Candidate peaks are ankle-separation maxima, not established contacts.',
               'Human video reference is not a force-platform measurement.',
               'Scoring must wait for reviewed reference and a fixed matching protocol.'],
    'videos': records, 'prediction_source_sha256': predictions_hash, 'image_manifest': images,
}
(ROOT / 'packet.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
(ROOT / 'candidates.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
# A separate file permits local HTML use without a server or an external dependency.
(ROOT / 'packet-data.js').write_text('window.PACKET = ' + json.dumps({k: v for k, v in packet.items() if k != 'image_manifest'}, ensure_ascii=False) + ';\n', encoding='utf-8')
(ROOT / 'candidate-data.js').write_text('window.CANDIDATES = ' + json.dumps(cases) + ';\n', encoding='utf-8')
assert digest(predictions_path) == predictions_hash
result = {'videos_checked': 2, 'source_frames_checked': 1022, 'lossless_review_frames': len(images),
          'cases': len(cases), 'reference_events_created': 0, 'accuracy_computed': False,
          'source_hashes_unchanged': True, 'opencv': cv2.__version__, 'python': sys.version.split()[0]}
(ROOT / 'preparation-checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
