"""Read original markers for display and simple gait measures. Scores are not changed here."""
import importlib.util
import json
import sys
import warnings
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('proj', ROOT / 'studies/20261103-reference188-projection-pilot/project.py')
proj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proj)
np = proj.np
ezc3d = proj.ezc3d
TMP = Path('D:/gait-research-data/D_P/older-normal100-20261106/current.c3d')
SHOW = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'LSHO', 'RSHO', 'LELB', 'RELB', 'LWRA', 'RWRA', 'LWRB', 'RWRB',
        'LASI', 'RASI', 'LPSI', 'RPSI', 'SACR', 'LKNE', 'RKNE', 'LANK', 'RANK', 'LHEE', 'RHEE', 'LTOE', 'RTOE',
        'LFEP', 'RFEP', 'LFEO', 'RFEO', 'LTIO', 'RTIO']
FOCUS = {'TVC4', 'SUBJ5'}


def read_trial(path_zip, member):
    with zipfile.ZipFile(path_zip) as z:
        TMP.write_bytes(z.read(member))
    c = ezc3d.c3d(str(TMP))
    labels = [n.split(':')[-1].strip() for n in c['parameters']['POINT']['LABELS']['value']]
    rate = float(c['header']['points']['frame_rate'])
    pts = c['data']['points'][:3].transpose(2, 1, 0) / 1000
    res = c['data']['meta_points']['residuals'][0].T
    pts[res < 0] = np.nan
    return labels, rate, pts


def fill(v):
    for j in range(v.shape[1]):
        for k in range(3):
            y = v[:, j, k]
            ok = np.isfinite(y)
            if ok.sum() >= 2 and not ok.all():
                y[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), y[ok])
    return v


def angle(a, b, c):
    u = a - b
    w = c - b
    cosv = np.sum(u * w, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
    return 180 - np.degrees(np.arccos(np.clip(cosv, -1, 1)))


def trial(labels, rate, pts):
    names = [n for n in SHOW if n in labels]
    x = fill(pts[:, [labels.index(n) for n in names]])
    t = np.arange(int((len(x) - 1) / rate * 30) + 1) / 30
    at = t * rate
    lo = np.floor(at).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (at - lo)[:, None, None]
    x = x[lo] * (1 - w) + x[hi] * w
    idx = {n: i for i, n in enumerate(names)}
    pelvis = (x[:, idx['LASI']] + x[:, idx['RASI']]) / 2
    d = pelvis[-1] - pelvis[0]
    d[2] = 0
    forward = d / np.linalg.norm(d)
    left = np.cross([0, 0, 1.], forward)
    local = (x - pelvis[0] * [1, 1, 0]) @ np.array([forward, left, [0, 0, 1.]]).T
    ground = np.nanpercentile(local[:, [idx[n] for n in names if n[1:] in ('TOE', 'HEE')], 2], 2)
    local[:, :, 2] -= ground
    p = (local[:, idx['LASI']] + local[:, idx['RASI']]) / 2
    speed = float(np.linalg.norm(p[-1, :2] - p[0, :2]) / t[-1])
    m = dict(speed=speed)
    for s, side in [('L', 'left'), ('R', 'right')]:
        k = angle(local[:, idx[s + 'FEP']], local[:, idx[s + 'FEO']], local[:, idx[s + 'TIO']])
        m[f'knee_{side}'] = float(np.nanmax(k) - np.nanmin(k))
        a = local[:, idx[s + 'TIO'], 0] - p[:, 0]
        m[f'ankle_{side}'] = float((np.nanmax(a) - np.nanmin(a)) * 100)
    m['knee_mean'] = (m['knee_left'] + m['knee_right']) / 2
    m['knee_gap'] = abs(m['knee_left'] - m['knee_right'])
    m['ankle_mean'] = (m['ankle_left'] + m['ankle_right']) / 2
    m['ankle_gap'] = abs(m['ankle_left'] - m['ankle_right'])
    return names, local, m


scores = json.loads((HERE / 'trial-scores.json').read_text(encoding='utf-8'))
people = {r['id']: r for r in json.loads((HERE / 'people-scores.json').read_text(encoding='utf-8'))}
measures = {}
focus = {}
for r in scores:
    if people[r['person']]['age'] < 65:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        labels, rate, pts = read_trial(r['archive'], r['member'])
        names, local, m = trial(labels, rate, pts)
    measures.setdefault(r['person'], []).append(m)
    if r['person'] in FOCUS:
        focus.setdefault(r['person'], []).append(dict(
            trial=r['member'].split('/')[-1], score_added=r['added'], score_baseline=r['baseline'], windows=r['windows'],
            names=names, frames=np.round(local[:, :, [0, 2]] * 1000).astype(int).tolist(), measures=m))
person_m = {k: {q: float(np.mean([t[q] for t in v])) for q in v[0]} for k, v in measures.items()}
out = dict(people=[dict(people[k], **{'gait': person_m[k]}) for k in person_m], focus=focus)
(HERE / 'motion-data.json').write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print({k: {a: round(b, 2) for a, b in person_m[k].items() if a in ('speed', 'knee_mean', 'knee_gap', 'ankle_mean', 'ankle_gap')} for k in FOCUS})
print({k: [t['trial'] + ' ' + str(round(t['score_added'], 1)) + ' ' + str(len(t['frames'])) + ' ' + str(len(t['names'])) for t in v] for k, v in focus.items()})
for c in ('healthy', 'stroke'):
    rows = [person_m[k] for k in person_m if people[k]['cohort'] == c]
    print(c, len(rows), {q: round(float(np.median([r[q] for r in rows])), 2) for q in ('speed', 'knee_mean', 'knee_gap', 'ankle_mean', 'ankle_gap')})
