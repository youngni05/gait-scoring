"""Record frame count, size and rate of each downloaded video."""
import json
from pathlib import Path
import cv2

HERE = Path('D:/Projects/D_P/studies/20261107-live-patient-eval')
VIDEOS = Path('D:/gait-research-data/D_P/live-patient-eval-20261107/videos')
rows = json.loads((HERE / 'download.json').read_text())['videos']
out = []
for r in rows:
    cap = cv2.VideoCapture(str(VIDEOS / r['file']))
    frames = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        frames += 1
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    cap.release()
    out.append(dict(file=r['file'], group=r['group'], person=r['person'], width=w, height=h, fps=round(fps, 3), frames=frames))
(HERE / 'video_metadata.json').write_text(json.dumps(out, indent=1))
print(len(out), sum(r['frames'] for r in out), sorted({(r['width'], r['height']) for r in out}), sorted({r['fps'] for r in out})[:6])
