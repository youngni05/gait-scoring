"""Replay completed adult64 models; no training or calibration on uploaded videos."""
import json
from functools import lru_cache

import engine

np = engine.np
MANIFEST_PATH = engine.HERE / 'comparison-models.json'
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=1)
def runtime():
    existing = engine.module('meeting_torch_inference', engine.ROOT /
        'gdi-research-harness/studies/20261101-self-video-gdi/predict.py')
    existing.torch.set_num_threads(2)
    return existing


def raw_member(member, x):
    path = engine.Path(member['path'])
    if engine.sha(path) != member['sha256']:
        raise ValueError('추가 학습 모델 파일 지문이 다르다.')
    r = runtime()
    checkpoint = r.torch.load(path, map_location='cpu', weights_only=True)
    identity = checkpoint['identity']
    if (identity['family'] != member['family'] or identity['outer_fold'] != member['fold']
            or identity['arm'] != 'Stanford_plus_LJMU100'):
        raise ValueError('추가 학습 모델의 출처가 맞지 않는다.')
    model = r.new_model(member['family'])
    model.load_state_dict(checkpoint['model'], strict=True)
    raw = r.forward(model, member['family'], x, checkpoint['lo'], checkpoint['span'])
    if raw.shape != (len(x), 2) or not np.isfinite(raw).all():
        raise ValueError('추가 학습 모델 출력이 올바르지 않다.')
    return raw


def fold_scores(values, manifest, fold):
    selected = next(s for s in manifest['selections'] if s['outer'] == fold)
    means = {family: np.mean([values[m['id']] for m in manifest['members']
             if m['family'] == family and m['fold'] == fold and m.get('inBag', True)], axis=0)
             for family in ('CNN', 'STT', 'bilateral')}
    base = selected['cnn_weight'] * means['CNN'] + (1-selected['cnn_weight']) * means['STT']
    delta = ((1-selected['fraction']) * (base[0]-base[1])
             + selected['fraction'] * (means['bilateral'][0]-means['bilateral'][1]))
    return {'CNN': values[f'CNN_{fold}_refit'], 'STT': values[f'STT_{fold}_refit'],
            'fusion': np.array([base.mean()+delta/2, base.mean()-delta/2])}


def summarize(values, manifest):
    folds = [fold_scores(values, manifest, f) for f in range(5)]
    return {kind: np.mean([f[kind] for f in folds], axis=0) for kind in ('CNN','STT','fusion')}


def compare(x, baseline_values, progress):
    mixed_values = {}
    total = len(engine.MANIFEST['members']) + len(MANIFEST['members'])
    for i, m in enumerate(MANIFEST['members']):
        raw = raw_member(m, x)
        mixed_values[m['id']] = raw.mean(0) * m['calibration'][0] + m['calibration'][1]
        progress(stage='score', done=54+i+1, total=total, message='정상 성인 자료 추가 학습 모델 계산 중')
    rows = []
    for arm, values, manifest in [('baseline', baseline_values, engine.MANIFEST), ('adult64', mixed_values, MANIFEST)]:
        for kind, value in summarize(values, manifest).items():
            rows.append(dict(id=f'{arm}_{kind}', arm=arm, model=kind,
                             left=float(value[0]), right=float(value[1]), mean=float(value.mean())))
    return dict(rows=rows, manifestSHA256=engine.sha(MANIFEST_PATH),
                normalPeople=64, normalRecords=881,
                aggregation='Same windows; CNN/STT: mean of five refit models. Fusion: frozen per-fold bag/calibration/weights, then five-fold mean.',
                clinicalValidation=False)
