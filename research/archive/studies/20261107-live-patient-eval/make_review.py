"""Record the visual review decisions. They were made before any model saw these videos."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
qc = json.loads((HERE / 'candidates-qc.json').read_text(encoding='utf-8'))
good = [d for d in qc if d['passes']]
reject = {
    'P18-240618-T2-GAIT-01-SAGITTAL': 'Both candidates rejected: c0 contains a turn toward the camera; c1 starts with a forward bend and possible hop before walking.',
    'P41-250206-T0-GAIT-01-SAGITTAL': 'Another person crosses in front of the camera and hides the target mid-window. Only candidate.',
}
accepted = {}
for rec in sorted({d['record'] for d in good}):
    if rec in reject:
        continue
    c = [d for d in good if d['record'] == rec][0]
    device = None
    if rec.startswith('P37-'):
        device = 'rolling walker'
    elif rec.startswith('P15-240604'):
        device = 'knee orthosis-like device'
    accepted[rec] = dict(candidate=c['candidate'], start=c['start'], moving=c['moving'], missing=c['missing'],
                         group=c['group'], person=c['person'], assistive_device=device)
out = dict(reviewed_by='Claude, 17-frame strips, before any model or classifier was run on these videos',
           date='2026-11-07', accepted=accepted, rejected=reject)
(HERE / 'review.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(len(accepted), len({v['person'] for v in accepted.values()}),
      {g: (sum(v['group'] == g for v in accepted.values()), len({v['person'] for v in accepted.values() if v['group'] == g})) for g in ('CP', 'ITW')})
