"""Post-hoc: on-screen hip movement in the self-recorded videos, same measure as posthoc_camera.py."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BIG = Path('D:/gait-research-data/D_P/self-video-test01-20261101')
T = Path('D:/Projects/D_P/gdi-research-harness/studies/20261101-self-video-test01')
windows = json.loads((T / 'windows.json').read_text(encoding='utf-8-sig'))
inv = json.loads((T / 'inventory.json').read_text(encoding='utf-8-sig'))
print(type(inv), (list(inv)[:6] if isinstance(inv, dict) else inv[0]))
out = {}
for vid in sorted({w['id'] for w in windows}):
    with np.load(BIG / f'{vid}-raw.npz', allow_pickle=False) as z:
        keys = z.files
        raw = z[keys[0]].astype(float)
    a = raw.reshape(-1, 25, 3)
    spans, speeds = [], []
    for w in [x for x in windows if x['id'] == vid]:
        s = a[w['start']:w['end'] + 1]
        ok = (s[:, 9, 2] > .1) & (s[:, 12, 2] > .1)
        h = ((s[:, 9, :2] + s[:, 12, :2]) / 2)[ok]
        th = (np.linalg.norm(s[:, 9, :2] - s[:, 10, :2], axis=1) + np.linalg.norm(s[:, 12, :2] - s[:, 13, :2], axis=1)) / 2
        t = np.nanmedian(th[ok])
        spans.append(np.ptp(h[:, 0]) / t)
        speeds.append(np.nanmedian(np.abs(np.diff(h[:, 0]))) / t)
    out[vid] = dict(number=windows[[x['id'] for x in windows].index(vid)]['number'], windows=len(spans), span_thigh=round(float(np.median(spans)), 2), speed=round(float(np.median(speeds)), 3))
vals = [v['span_thigh'] for v in out.values()]
print(json.dumps(out), keys, 'median span', np.median(vals), 'median speed', np.median([v['speed'] for v in out.values()]))
(HERE / 'posthoc-self-capture.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
