"""study-protocol, manifest, task list and upload archive for the continued-training arm."""
import hashlib, json, tarfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
B = HERE / 'bundle'
BIG = Path('D:/gait-research-data/D_P/finetune-20261110'); BIG.mkdir(parents=True, exist_ok=True)
DS = json.loads((HERE.parent / '20261110-dose-seed/bundle/study-protocol.json').read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
proto = {
    'date': '2026-11-10',
    'authorization': 'User asked on 2026-11-10 to run experiment C: continued training from the baseline model.',
    'question': 'Does short continued training of the published baseline members on Stanford plus LJMU100 recover normal-gait scores with less loss of evaluation accuracy than training from scratch?',
    'design': 'Same data, folds, seeds and ensemble weights as the published normal-added model. Each member starts from the baseline member of the same outer and inner fold, so its Stanford training people are identical and no validation or outer person was seen. Optimizer state starts fresh. Epoch chosen by the weighted mean of the available inner validation curves (Stanford and LJMU validation people, as in the published model). CNN fold 0 has three baseline inner members. Bilateral STT has only baseline refit members, so its refit uses the epoch selected for STT in the same fold.',
    'finetune': {'CNN': {'learning_rate': 0.0001, 'max_epochs': 30},
                 'STT': {'learning_rate': 3e-05, 'eta_min': 8e-06, 'max_epochs': 40},
                 'bilateral': {'learning_rate': 3e-05, 'eta_min': 8e-06, 'max_epochs': 40}},
    'calibration_for_reporting': 'Measured-value calibration on Stanford training rows, as in the paper.',
    'external_sets': DS['external_sets'],
    'criteria_fixed_in_advance': {
        'accuracy': 'Evaluation-set person MAE change against the baseline ensemble smaller than the published normal-added change (+0.126), reported with a 95 percent person bootstrap interval.',
        'normal_scores': 'LIVE typically developing and self-captured normal means at least 90.',
        'selectivity': 'Normal minus abnormal gap larger than the baseline in all three external sources.'},
    'data_sha256': DS['data_sha256'], 'external_sha256': DS['external_sha256'],
}
(B / 'study-protocol.json').write_text(json.dumps(proto, indent=1) + '\n', encoding='utf-8')
files = sorted(p for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json', 'approved.json'))
(B / 'manifest.json').write_text(json.dumps({p.relative_to(B).as_posix(): sha(p) for p in files}, indent=1) + '\n', encoding='utf-8')
with tarfile.open(BIG / 'bundle.tar.gz', 'w:gz') as t:
    for p in files + [B / 'manifest.json']:
        t.add(p, arcname=p.relative_to(B).as_posix())
meta = {'bundle_tar_sha256': sha(BIG / 'bundle.tar.gz'), 'manifest_sha256': sha(B / 'manifest.json')}
(HERE / 'preparation.json').write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
print(meta)
