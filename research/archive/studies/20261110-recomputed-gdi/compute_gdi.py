"""Gait vectors and recomputed GDI from Plug-in Gait C3D (Van Criekinge et al. 2023). No model, no outcomes."""
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np

sys.path[:0] = ['D:/gait-research-data/D_P/dataset-audits/20261103-reference188-projection-pilot/deps',
                'D:/Projects/D_P/studies/20261103-reference188-pilot/deps',
                'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps']
import ezc3d

HERE = Path(__file__).resolve().parent
SRC = Path('D:/gait-research-data/D_P/frozen-transfer-and-deviation-20261103')
BIG = Path('D:/gait-research-data/D_P/recomputed-gdi-20261110')
BIG.mkdir(parents=True, exist_ok=True)
CURVES = [('PelvisAngles', 0), ('PelvisAngles', 1), ('PelvisAngles', 2), ('HipAngles', 0), ('HipAngles', 1), ('HipAngles', 2),
          ('KneeAngles', 0), ('AnkleAngles', 0), ('FootProgressAngles', 2)]


def person_of(name):
    m = re.search(r'/(SUBJ|TVC)0*(\d+)/', name, re.I)
    return None if m is None else (m.group(1).upper() + str(int(m.group(2))))


def cycles(path):
    c = ezc3d.c3d(path)
    rate = float(c['parameters']['POINT']['RATE']['value'][0])
    first = int(c['header']['points']['first_frame'])
    labels = [l.split(':')[-1].strip() for l in c['parameters']['POINT']['LABELS']['value']]
    pts = c['data']['points'][:3]
    ev = c['parameters'].get('EVENT', {})
    if not ev or 'TIMES' not in ev:
        return {}
    ctx = [s.strip() for s in ev['CONTEXTS']['value']]
    lab = [s.strip() for s in ev['LABELS']['value']]
    times = np.array(ev['TIMES']['value'])[1]
    out = {}
    for side, s in (('Left', 'L'), ('Right', 'R')):
        names = [s + n for n, _ in CURVES]
        if not all(n in labels for n in names):
            continue
        arr = np.stack([pts[k, labels.index(s + n), :] for n, k in CURVES])  # 9 x frames (points axis is the label)
        fs = sorted(t for t, cx, lb in zip(times, ctx, lab) if cx == side and lb == 'Foot Strike')
        idx = [int(round(t * rate)) - first for t in fs]
        res = []
        for a, b in zip(idx[:-1], idx[1:]):
            dur = (b - a) / rate
            if not (0.6 <= dur <= 2.5) or a < 0 or b >= arr.shape[1]:
                continue
            seg = arr[:, a:b + 1]
            if not np.isfinite(seg).all() or (np.abs(seg).sum(0) == 0).any():
                continue
            t0 = np.linspace(0, 1, seg.shape[1])
            res.append(np.concatenate([np.interp(np.linspace(0, 1, 51), t0, row) for row in seg]))
        if res:
            out[s] = res
    return out


vectors = {}
counts = {'files': 0, 'files_with_cycles': 0, 'cycles': 0}
tmp = os.path.join(tempfile.gettempdir(), 'gdi-recompute.c3d')
for zname in ('138_HealthyPiG.zip', '50_StrokePiG.zip'):
    z = zipfile.ZipFile(SRC / zname)
    for name in z.namelist():
        if not name.lower().endswith('.c3d'):
            continue
        person = person_of('/' + name)
        if person is None:
            continue
        counts['files'] += 1
        Path(tmp).write_bytes(z.read(name))
        try:
            cyc = cycles(tmp)
        except Exception as e:  # unreadable file is counted, not hidden
            counts.setdefault('read_errors', []).append(f'{name}: {type(e).__name__}')
            continue
        if cyc:
            counts['files_with_cycles'] += 1
        for s, lst in cyc.items():
            vectors.setdefault((person, s), []).extend(lst)
            counts['cycles'] += len(lst)


LINK = json.loads(Path('D:/Projects/D_P/studies/20261103-reference188-pilot/person-linkage.json').read_text(encoding='utf-8-sig'))
meta = {}
for g in ('healthy', 'stroke'):
    for row in LINK[g]['rows']:
        m = re.fullmatch(r'(SUBJ|TVC)0*(\d+)', row['supplement_candidate_ids'][0].upper())
        meta[m[1] + str(int(m[2]))] = {'age': row['values']['Age'], 'lesion_left': row['values'].get('LesionLeft')}
keys = sorted(vectors)
G = np.array([np.mean(vectors[k], axis=0) for k in keys])
person = np.array([k[0] for k in keys])
side = np.array([k[1] for k in keys])
cohort = np.array(['healthy' if p.startswith('SUBJ') else 'stroke' for p in person])
age = np.array([meta[p]['age'] if p in meta else np.nan for p in person], dtype=float)
ncyc = np.array([len(vectors[k]) for k in keys])
paretic = np.array([(meta[p]['lesion_left'] == 1 and s == 'R') or (meta[p]['lesion_left'] == 0 and s == 'L') if p in meta and meta[p]['lesion_left'] is not None else False for p, s in zip(person, side)])

# features: SVD of the uncentred matrix of all limb vectors
U, S, Vt = np.linalg.svd(G, full_matrices=False)
F = G @ Vt[:15].T
explained = float((S[:15] ** 2).sum() / (S ** 2).sum())


def gdi(ref_mask):
    out = np.full(len(F), np.nan)
    for i in range(len(F)):
        ref = ref_mask & (person != person[i])
        mu = F[ref].mean(0)
        lr = np.log(np.linalg.norm(F[ref] - mu, axis=1))
        # reference ln-distances are computed against the same reference mean
        z = (np.log(np.linalg.norm(F[i] - mu)) - lr.mean()) / lr.std(ddof=1)
        out[i] = 100 - 10 * z
    return out


R = {'R_all': cohort == 'healthy', 'R_young': (cohort == 'healthy') & (age < 65)}
scores = {k: gdi(m) for k, m in R.items()}
np.savez_compressed(BIG / 'gait-vectors.npz', G=G, person=person, side=side, cohort=cohort, age=age, cycles=ncyc, paretic=paretic, **{f'gdi_{k}': v for k, v in scores.items()})
summary = {'counts': {k: v for k, v in counts.items() if k != 'read_errors'}, 'read_errors': len(counts.get('read_errors', [])),
           'limbs': int(len(keys)), 'people': {c: int(len(set(person[cohort == c]))) for c in ('healthy', 'stroke')},
           'healthy_young_people': int(len(set(person[(cohort == 'healthy') & (age < 65)]))),
           'missing_age': int(np.isnan(age).sum()), 'svd_15_variance_share': round(explained, 4),
           'reference_sizes_limbs': {k: int(m.sum()) for k, m in R.items()}}
for k, v in scores.items():
    for c in ('healthy', 'stroke'):
        for band, m in (('all', np.ones(len(v), bool)), ('ge65', age >= 65), ('lt65', age < 65)):
            sel = (cohort == c) & m
            if sel.any():
                summary.setdefault(k, {})[f'{c}_{band}'] = [round(float(v[sel].mean()), 2), int(sel.sum())]
summary['gait_vectors_sha256'] = hashlib.sha256((BIG / 'gait-vectors.npz').read_bytes()).hexdigest()
(HERE / 'gdi-summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=1, ensure_ascii=False))
