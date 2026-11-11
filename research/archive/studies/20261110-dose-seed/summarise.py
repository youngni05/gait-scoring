"""Combine the arm summaries with the published s0 endpoints and apply the criteria fixed in study-protocol.json."""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('aa', HERE / 'analyze_arms.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
REC = Path('D:/gait-research-data/D_P/dose-seed-20261110/received/runs')
ST = HERE.parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


arms = {k: A.summarise(REC / k) for k in ['n0_s1', 'n0_s2', 'n64_s1', 'n64_s2', 'n4_s0', 'n8_s0', 'n16_s0', 'n32_s0']}
# published s0 endpoints (same measured-value calibration and ensemble), read from their result files
live = read(ST / '20261107-live-patient-eval/results.json')
older = read(ST / '20261106-older-normal100/results.json')['arms']
sb = read(ST / '20261105-unseen-normal-test/results.json')['self']['baseline']
sa = read(ST / '20261105-unseen-normal-test/stanford-calibration/results.json')['external']['mixed_stanford_cal']
fin = read(ST / '20261106-final222-normal100/results.json')['models']['fusion']['all']


def pub(before):
    m = {'LIVE_TD': live['TD']['before' if before else 'after_measured']['mean'],
         'LIVE_CP': live['groups']['CP']['before' if before else 'after_measured']['mean'],
         'self_normal': sb['normal_mean'] if before else sa['self_normal_mean'],
         'self_abnormal': sb['abnormal_mean'] if before else sa['self_abnormal_mean'],
         'older_healthy': older['baseline' if before else 'stan']['healthy65']['mean'],
         'older_stroke': older['baseline' if before else 'stan']['stroke65']['mean']}
    m = {k: round(float(v), 2) for k, v in m.items()}
    return {'eval_mae': round(float(fin['baseline']['mae'] if before else fin['mixed_stanford_cal']['mae']), 3), 'means': m,
            'gaps': {'LIVE': round(m['LIVE_TD'] - m['LIVE_CP'], 2), 'self': round(m['self_normal'] - m['self_abnormal'], 2),
                     'older': round(m['older_healthy'] - m['older_stroke'], 2)}}


arms['n0_s0'] = pub(True)
arms['n64_s0'] = pub(False)
dose = ['n0_s0', 'n4_s0', 'n8_s0', 'n16_s0', 'n32_s0', 'n64_s0']
people = [0, 4, 8, 16, 32, 64]
rk = lambda v: np.argsort(np.argsort(v))
crit = {}
for g in ('LIVE_TD', 'self_normal'):
    v = [arms[a]['means'][g] for a in dose]
    crit[f'dose_spearman_{g}'] = round(float(np.corrcoef(rk(people), rk(v))[0, 1]), 3)
crit['dose_gap_ge16_larger_than_0'] = all(arms[a]['gaps'][s] > arms['n0_s0']['gaps'][s] for a in ('n16_s0', 'n32_s0', 'n64_s0') for s in ('LIVE', 'self', 'older'))
crit['dose_pass'] = crit['dose_spearman_LIVE_TD'] > 0 and crit['dose_spearman_self_normal'] > 0 and crit['dose_gap_ge16_larger_than_0']
seed = {}
for s in ('s0', 's1', 's2'):
    b, a = arms[f'n0_{s}'], arms[f'n64_{s}']
    seed[s] = {'LIVE_TD_rise': round(a['means']['LIVE_TD'] - b['means']['LIVE_TD'], 2),
               'gap_change': {k: round(a['gaps'][k] - b['gaps'][k], 2) for k in ('LIVE', 'self', 'older')},
               'eval_mae_change': round(a['eval_mae'] - b['eval_mae'], 3)}
crit['seed_same_sign'] = all(v['LIVE_TD_rise'] > 0 and all(x > 0 for x in v['gap_change'].values()) and v['eval_mae_change'] > 0 for v in seed.values())
counts = {0: 0, 4: 62, 8: 121, 16: 233, 32: 450, 64: 881}
out = {'arms': arms, 'dose_order': dose, 'ljmu_people': people, 'ljmu_videos': [counts[p] for p in people], 'seed': seed, 'criteria': crit}
(HERE / 'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'criteria': crit, 'seed': seed}, ensure_ascii=False, indent=1))
for a, p in zip(dose, people):
    r = arms[a]
    print(p, r['eval_mae'], r.get('eval_delta_mae_ci95'), r['means']['LIVE_TD'], r['means']['self_normal'], r['means']['older_healthy'], r['means']['LIVE_CP'], r['gaps'])
