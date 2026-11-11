"""Post-hoc, not preregistered: does the CP-TD rise difference remain within the TD age range?"""
import json
from pathlib import Path
import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
spec = {'__file__': str(HERE / 'analyze.py')}
exec(open(HERE / 'analyze.py', encoding='utf-8').read().split("rng = np.random.default_rng")[0], spec)
tdp, tdid = spec['tdp'], spec['tdid']
oldmeta = spec['oldmeta']
meta = spec['meta']
wb = openpyxl.load_workbook('D:/Projects/D_P/gdi-research-harness/studies/20261031-cnn-high-gdi/research/PARTICIPANT_CHARACTERISTICS.xlsx', read_only=True, data_only=True)
table = {}
for g in ('CP', 'ITW'):
    for row in list(wb[g].values)[1:]:
        if row[1] and isinstance(row[4], (int, float)):
            table[(row[1], row[2])] = (float(row[4]), row[17])
age, dx, missing = {}, {}, []
for r in meta['records']:
    phase = r['record'].split('-')[2]
    hit = table.get((r['person'], phase))
    if hit is None:
        hit = next((v for k, v in table.items() if k[0] == r['person']), None)
        missing.append(r['record'])
    age.setdefault(r['person'], []).append(hit[0])
    dx[r['person']] = hit[1]
age = {k: float(np.mean(v)) for k, v in age.items()}
tdage = dict(zip(tdid.tolist(), oldmeta['live_ages']))
tdq = sorted(set(tdid.tolist()))
tdrise = tdp['after_measured'] - tdp['before']
tda = np.array([tdage[q] for q in tdq])
lim = float(tda.max())
rng = np.random.default_rng(20261107)
B = 10000
tddraw = rng.integers(0, len(tdrise), (B, len(tdrise)))


def compare(g, keep):
    v = R['groups'][g]['per_person']
    q = [k for k in v if keep(k.split(':')[1])]
    rise = np.array([v[k]['after_measured'] - v[k]['before'] for k in q])
    after = np.array([v[k]['after_measured'] for k in q])
    before = np.array([v[k]['before'] for k in q])
    draw = rng.integers(0, len(q), (B, len(q)))
    d = rise[draw].mean(1) - tdrise[tddraw].mean(1)
    return dict(people=len(q), persons=[k.split(':')[1] for k in q], mean_age=float(np.mean([age[k.split(':')[1]] for k in q])),
                before=float(before.mean()), after=float(after.mean()), rise=float(rise.mean()),
                rise_minus_TD=float(rise.mean() - tdrise.mean()), ci95=[float(x) for x in np.quantile(d, [.025, .975])])


out = dict(td_age_range=[float(tda.min()), lim], td_mean_age=float(tda.mean()), td_rise=float(tdrise.mean()), records_phase_not_found=len(missing))
out['CP_all'] = compare('CP', lambda p: True)
out['CP_within_TD_age'] = compare('CP', lambda p: age[p] <= lim)
out['CP_older'] = compare('CP', lambda p: age[p] > lim)
out['CP_within_TD_age_diparesis'] = compare('CP', lambda p: age[p] <= lim and 'Dipar' in dx[p])
out['CP_within_TD_age_unilateral'] = compare('CP', lambda p: age[p] <= lim and 'Dipar' not in dx[p])
out['ITW_within_TD_age'] = compare('ITW', lambda p: age[p] <= lim)
v = R['groups']['CP']['per_person']
a = np.array([age[k.split(':')[1]] for k in v])
r = np.array([x['after_measured'] - x['before'] for x in v.values()])
out['CP_rise_age_pearson'] = float(np.corrcoef(a, r)[0, 1])
(HERE / 'posthoc-age.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
print(json.dumps(out, indent=0))
