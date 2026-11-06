"""Save person-level scores for the fixed older-cohort comparison. No new fitting."""
import json

from analyze import HERE, P, clin, people

out = []
for k in P['baseline']:
    p = people[k]
    c = clin.get(k)
    out.append(dict(id=k, cohort=p['cohort'], age=p['age'], baseline=P['baseline'][k], added=P['stan'][k],
                    added_mixed_calibration=P['mixed'][k], POMA=c['POMA'] if c else None, FAC=c['FAC'] if c else None,
                    TIS=c['TIS'] if c else None, speed_mps=c['speed']['mean_mps'] if c else None))
(HERE / 'people-scores.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
old = sorted((r for r in out if r['age'] >= 65), key=lambda r: (r['cohort'], r['added']))
for r in old:
    print(r['cohort'], r['id'], r['age'], round(r['baseline'], 1), round(r['added'], 1), r['POMA'], r['FAC'], r['speed_mps'])
