"""Paretic versus other leg: model scores against recomputed limb GDI. No new inference."""
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OLDER = HERE.parent / '20261106-older-normal100'
# analyze.py rewrites its own results.json when imported; keep a copy and confirm it is unchanged afterwards
keep = Path(tempfile.gettempdir()) / 'older-results-before-import.json'
shutil.copyfile(OLDER / 'results.json', keep)
spec = importlib.util.spec_from_file_location('older_analyze', OLDER / 'analyze.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
assert (OLDER / 'results.json').read_bytes() == keep.read_bytes(), 'older-normal100 results.json changed'

legs = {}
for r, row in enumerate(A.rows):
    ix = np.flatnonzero(A.owner == r)
    if row['primary'] and len(ix):
        legs.setdefault(row['person'], {'baseline': [], 'added': []})
        legs[row['person']]['baseline'].append(A.pred['baseline'][ix])
        legs[row['person']]['added'].append(A.pred['stan'][ix])
model = {p: {k: np.concatenate(v[k]).mean(0) for k in v} for p, v in legs.items()}  # [left, right]

z = np.load('D:/gait-research-data/D_P/recomputed-gdi-20261110/gait-vectors.npz')
link = json.loads(Path('D:/Projects/D_P/studies/20261103-reference188-pilot/person-linkage.json').read_text(encoding='utf-8-sig'))
lesion = {}
import re
for row in link['stroke']['rows']:
    m = re.fullmatch(r'(SUBJ|TVC)0*(\d+)', row['supplement_candidate_ids'][0].upper())
    lesion[m[1] + str(int(m[2]))] = row['values'].get('LesionLeft')
gdi = {}
for p, s, g in zip(z['person'], z['side'], z['gdi_R_all']):
    gdi.setdefault(str(p), {})[str(s)] = float(g)

rng = np.random.default_rng(20261110)
stroke = [p for p in model if p.startswith('TVC') and p in gdi and len(gdi[p]) == 2 and lesion.get(p) in (0, 1)]
rows = []
for p in stroke:
    par = 1 if lesion[p] == 1 else 0  # left lesion -> right leg (index 1) paretic
    oth = 1 - par
    sides = ['L', 'R']
    rows.append({'person': p, 'paretic_side': sides[par],
                 'recomputed': gdi[p][sides[par]] - gdi[p][sides[oth]],
                 'baseline': float(model[p]['baseline'][par] - model[p]['baseline'][oth]),
                 'added': float(model[p]['added'][par] - model[p]['added'][oth])})
D = {k: np.array([r[k] for r in rows]) for k in ('recomputed', 'baseline', 'added')}
n = len(rows)
draw = rng.integers(0, n, (10000, n))
q = lambda v: [round(float(t), 3) for t in np.quantile(v, [.025, .975])]
out = {'stroke_people': n, 'paretic_right': int(sum(r['paretic_side'] == 'R' for r in rows))}
for k, v in D.items():
    low = (v < 0).astype(float)
    out[k] = {'paretic_lower_share': round(float(low.mean()), 3), 'paretic_lower_share_ci95': q(low[draw].mean(1)),
              'mean_paretic_minus_other': round(float(v.mean()), 2), 'mean_ci95': q(v[draw].mean(1))}
for k in ('baseline', 'added'):
    out[k]['corr_with_recomputed'] = round(float(np.corrcoef(D[k], D['recomputed'])[0, 1]), 3)
    cs = [np.corrcoef(D[k][d], D['recomputed'][d])[0, 1] for d in draw[:4000]]
    out[k]['corr_ci95'] = q(np.array(cs))
    out[k]['sign_agreement_with_recomputed'] = round(float((np.sign(D[k]) == np.sign(D['recomputed'])).mean()), 3)
healthy = [p for p in model if p.startswith('SUBJ') and p in gdi and len(gdi[p]) == 2]
out['healthy_abs_left_right'] = {'people': len(healthy),
                                 'recomputed': round(float(np.mean([abs(gdi[p]['L'] - gdi[p]['R']) for p in healthy])), 2),
                                 'baseline': round(float(np.mean([abs(model[p]['baseline'][0] - model[p]['baseline'][1]) for p in healthy])), 2),
                                 'added': round(float(np.mean([abs(model[p]['added'][0] - model[p]['added'][1]) for p in healthy])), 2)}
(HERE / 'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
(HERE / 'people.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(out, indent=1, ensure_ascii=False))
