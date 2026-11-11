"""Build external.npz, study-protocol.json, manifest.json, tasks.txt and the upload archive. CPU only, no training."""
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / 'bundle'
BIG = Path('D:/gait-research-data/D_P/dose-seed-20261110')
BIG.mkdir(parents=True, exist_ok=True)
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104/data.npz')
FINAL = Path('D:/gait-research-data/D_P/final223-frozen-20261102/features.npz')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105/inputs.npz')
PAT = Path('D:/gait-research-data/D_P/live-patient-eval-20261107/inputs.npz')
OLDER = Path('D:/gait-research-data/D_P/frozen-transfer-and-deviation-20261103/features.npz')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ext = {}
with np.load(FINAL) as z:
    ext.update(eval__x=z['x'], eval__owner=z['window_record'], eval__n=np.int64(len(z['person_id'])))
with np.load(OLD) as z:
    ext.update(livetd__x=z['live__x'], livetd__owner=z['live__owner'], livetd__n=np.int64(z['live__n']),
               self__x=z['self__x'], self__owner=z['self__owner'], self__n=np.int64(z['self__n']))
with np.load(PAT) as z:
    ext.update(livepat__x=z['x'], livepat__owner=z['owner'], livepat__n=np.int64(z['owner'].max() + 1))
with np.load(OLDER) as z:
    x = z['x']
    ext.update(older__x=x, older__owner=np.arange(len(x)), older__n=np.int64(len(x)))
for k in list(ext):
    if k.endswith('__x'):
        ext[k] = np.ascontiguousarray(ext[k], dtype=np.float32)
        assert ext[k].shape[1:] == (2, 124, 12) and np.isfinite(ext[k]).all(), k
        n = int(ext[k[:-3] + '__n'])
        assert np.array_equal(np.unique(ext[k[:-3] + '__owner']), np.arange(n)), k
np.savez_compressed(BIG / 'external.npz', **ext)
sets = ['eval', 'livetd', 'self', 'livepat', 'older']
print({s: ext[f'{s}__x'].shape[0] for s in sets})

arms = {'n4_s0': (4, 0), 'n8_s0': (8, 0), 'n16_s0': (16, 0), 'n32_s0': (32, 0),
        'n0_s1': (0, 1000), 'n0_s2': (0, 2000), 'n64_s1': (64, 1000), 'n64_s2': (64, 2000)}
proto = {
    'date': '2026-11-10',
    'authorization': 'User asked on 2026-11-10 to run the long GPU experiments first (dose curve and seed repeats), with GPU jobs split to finish sooner.',
    'purpose': 'Dose: how normal-gait scores and evaluation accuracy change with the number of assumed-normal LJMU people, under the same design as the published normal-added model. Seeds: whether the published before/after differences hold under other seed schedules.',
    'design': 'Same frozen CNN, STT and bilateral-STT training as studies/20261104-ljmu100-mixed: five person folds, four inner fits plus a refit, epoch chosen by inner validation, frozen ensemble weights. LJMU people of an arm keep their original fold and all their records. People are chosen nested and fold-balanced by a fixed hash. Seed offset is added to the protocol seed of every family.',
    'existing_arms': {'n0_s0': 'published baseline models', 'n64_s0': 'published normal-added models'},
    'arms': {k: {'ljmu_people': v[0], 'seed_offset': v[1]} for k, v in arms.items()},
    'calibration_for_reporting': 'Linear calibration fitted on Stanford training rows only (measured-value calibration), as in the paper.',
    'external_sets': sets,
    'external_sources': {'eval': str(FINAL), 'livetd': str(OLD) + ' live__', 'self': str(OLD) + ' self__', 'livepat': str(PAT), 'older': str(OLDER)},
    'evaluation': 'CNN-STT ensemble per arm. Evaluation set: person MAE and correlation. External: mean score of LIVE typically developing, LIVE cerebral palsy, self-captured normal and simulated abnormal, healthy and stroke adults aged 65 or more; normal minus abnormal gap in each source.',
    'criteria_fixed_in_advance': {
        'dose': 'Rank correlation between LJMU people (0,4,8,16,32,64) and the normal score is positive for LIVE typically developing children and self-captured normal gait; the normal minus abnormal gap at 16 or more people is larger than at 0 in all three external sources.',
        'seeds': 'For each of the three seed schedules, normal-added minus baseline has the same sign as the published result for: LIVE typically developing score rise, normal minus abnormal gap change in all three sources, evaluation MAE change.'},
    'data_sha256': sha(SRC),
    'external_sha256': sha(BIG / 'external.npz'),
}
(BUNDLE / 'study-protocol.json').write_text(json.dumps(proto, indent=1) + '\n', encoding='utf-8')

# tasks: longest first (bilateral, CNN, STT), larger arms first
order = sorted(arms, key=lambda k: -arms[k][0])
tasks = [f'{arm} {fam} {f}' for fam in ('bilateral', 'CNN', 'STT') for arm in order for f in range(5)]
(BUNDLE / 'tasks.txt').write_text('\n'.join(tasks) + '\n', encoding='utf-8')

files = sorted(p for p in BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json', 'approved.json'))
manifest = {p.relative_to(BUNDLE).as_posix(): sha(p) for p in files}
(BUNDLE / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
with tarfile.open(BIG / 'bundle.tar.gz', 'w:gz') as t:
    for p in files + [BUNDLE / 'manifest.json']:
        t.add(p, arcname=p.relative_to(BUNDLE).as_posix())
meta = {'tasks': len(tasks), 'bundle_tar_sha256': sha(BIG / 'bundle.tar.gz'), 'external_sha256': proto['external_sha256'],
        'data_sha256': proto['data_sha256'], 'manifest_sha256': sha(BUNDLE / 'manifest.json')}
(HERE / 'preparation.json').write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
print(meta)
