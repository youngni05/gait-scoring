"""Post-hoc: image-level capture differences in development poses versus external poses. No model runs here."""
import json
import re
import sys
import zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
BIG = Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
ARC = Path('D:/gait-research-data/D_P/outputs/gait-public-access-20261015/Stanford/video-gait-v1.zip')
with np.load(SRC / 'data.npz') as z:
    d = {k: z[k] for k in ('window_record', 'window_start', 'y', 'source', 'video_id')}
n = int((d['source'] == 'Stanford').sum())
hi = np.flatnonzero((d['source'] == 'Stanford') & (d['y'].min(1) >= 90))
rng = np.random.default_rng(20261107)
rest = rng.choice(np.setdiff1d(np.arange(n), hi), 140, replace=False)
pick = {int(i): ('high' if i in set(hi.tolist()) else 'rest') for i in np.concatenate([hi, rest])}
lookup = {'openpose/' + str(d['video_id'][i]).split(':', 1)[1] + '-processed': i for i in pick}
A = zipfile.ZipFile(ARC)
names = A.namelist()
print(len(names), names[100:102], flush=True)
folders = {}
for m in names:
    f = m.rsplit('/', 1)[0]
    if f in lookup and m.endswith('.json'):
        folders.setdefault(f, []).append(m)
print(len(folders), flush=True)


def metrics(frames, width, height):
    """frames: list of 25x3 arrays or None."""
    a = np.array([f if f is not None else np.full((25, 3), np.nan) for f in frames])
    ok = (a[:, 9, 2] > .1) & (a[:, 12, 2] > .1)
    hip = (a[:, 9, :2] + a[:, 12, :2]) / 2
    th = (np.linalg.norm(a[:, 9, :2] - a[:, 10, :2], axis=1) + np.linalg.norm(a[:, 12, :2] - a[:, 13, :2], axis=1)) / 2
    th = th[ok & np.isfinite(th)]
    h = hip[ok]
    if len(h) < 30 or len(th) < 30:
        return None
    return dict(span_frac=float(np.ptp(h[:, 0]) / width), span_thigh=float(np.ptp(h[:, 0]) / np.median(th)), thigh_px=float(np.median(th)),
                thigh_frac=float(np.median(th) / height), thigh_change=float(np.quantile(th, .9) / np.quantile(th, .1)),
                speed_thigh_per_frame=float(np.nanmedian(np.abs(np.diff(h[:, 0]))) / np.median(th)))


rows = {'high': [], 'rest': []}
for f, items in folders.items():
    i = lookup[f]
    w = np.flatnonzero(d['window_record'] == i)
    start = int(d['window_start'][w[len(w) // 2]])
    items = sorted(items)[start:start + 124]
    frames = []
    for m in items:
        people = [np.asarray(p['pose_keypoints_2d'], float).reshape(25, 3) for p in json.loads(A.read(m))['people']]
        frames.append(people[0] if len(people) == 1 else None)
    r = metrics(frames, 320, 240)
    if r:
        rows[pick[i]].append(r)
out = {}
for k, v in rows.items():
    out['Stanford_' + k] = {q: round(float(np.median([r[q] for r in v])), 3) for q in v[0]} | {'records': len(v)}
pre = json.loads((HERE / 'preparation.json').read_text(encoding='utf-8'))
cand = {r['record']: r for r in json.loads((HERE / 'candidates.json').read_text(encoding='utf-8'))}
rev = json.loads((HERE / 'review.json').read_text(encoding='utf-8'))['accepted']
v = []
for r in pre['records']:
    c = cand[r['record']]['candidates'][rev[r['record']]['candidate']]
    fr = []
    for k, sel in enumerate(c['person_indices']):
        fr.append(None)
        if sel >= 0:
            p = json.loads((BIG / 'body25' / r['record'] / ('%s_%012d_keypoints.json' % (r['record'], c['start'] + k))).read_text())['people']
            fr[-1] = np.asarray(p[sel]['pose_keypoints_2d'], float).reshape(25, 3)
    m = metrics(fr, cand[r['record']]['width'], cand[r['record']]['height'])
    if m:
        v.append(m)
out['LIVE_patients_fixed_camera'] = {q: round(float(np.median([r[q] for r in v])), 3) for q in v[0]} | {'records': len(v)}
(HERE / 'posthoc-camera.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
print(json.dumps(out, indent=1))
