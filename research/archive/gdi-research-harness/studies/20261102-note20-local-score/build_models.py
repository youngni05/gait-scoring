"""Export the frozen 54 members; replay existing coordinate features without fitting."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BIG = Path('D:/gait-research-data/D_P/note20-score-20261102')
sys.path[:0] = [str(BIG / 'deps'), str(ROOT / 'outputs/motionbert-smoke-20261015/deps')]
import numpy as np
import onnx
import onnxruntime as ort
import torch

SOURCE = HERE.parent / '20261102-final223-frozen/models.json'
OLD = HERE.parent / '20261101-self-video-gdi'
spec = importlib.util.spec_from_file_location('phone_frozen_prior', OLD / 'predict.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class ExportMember(torch.nn.Module):
    def __init__(self, member, family, lo, span):
        super().__init__()
        self.member, self.family = member.eval(), family
        self.lo, self.span = float(lo), float(span)

    def forward(self, features):
        if self.family == 'CNN':
            value = self.member(features.permute(0, 1, 3, 2).reshape(-1, 12, 124))[:, 0]
        elif self.family == 'STT':
            value = self.member(features[..., :8].reshape(-1, 124, 4, 2))
        else:
            value = self.member(features[..., :8].reshape(-1, 2, 124, 4, 2))
        return (value * self.span + self.lo).reshape(-1, 2)


def combine(predictions, members, selections):
    folds = []
    for fold, selection in enumerate(selections):
        means = {}
        for family in ('CNN', 'STT', 'bilateral'):
            means[family] = np.mean([predictions[m['id']] for m in members
                                     if m['family'] == family and m['fold'] == fold], axis=0)
        base = selection['cnn_weight'] * means['CNN'] + (1 - selection['cnn_weight']) * means['STT']
        center = base.mean()
        difference = ((1 - selection['fraction']) * (base[0] - base[1])
                      + selection['fraction'] * (means['bilateral'][0] - means['bilateral'][1]))
        folds.append(np.array([center + difference / 2, center - difference / 2]))
    return np.mean(folds, axis=0)


def main():
    assets = BIG / 'assets'
    (assets / 'models').mkdir(parents=True, exist_ok=True)
    (assets / 'samples').mkdir(exist_ok=True)
    source = read(SOURCE)
    metadata = read(OLD / 'windows.json')
    old_results = read(OLD / 'results.json')
    feature_path = Path('D:/gait-research-data/D_P/self-video-gdi-20261101/features.npz')
    assert sha(feature_path) == old_results['features_sha256']
    with np.load(feature_path) as z:
        all_x = z['x'].copy()
    with np.load('D:/gait-research-data/D_P/self-video-gdi-20261101/predictions.npz') as z:
        saved = {k: z[k].copy() for k in z.files}
    samples = []
    for index, clip in enumerate(('clip0', 'clip1')):
        rows = [i for i, w in enumerate(metadata) if w['clip'] == clip and w['kind'] == 'primary']
        x = np.ascontiguousarray(all_x[rows], dtype='<f4')
        path = assets / 'samples' / ('sample-%d.f32' % index)
        x.tofile(path)
        samples.append(dict(id='sample-%d' % index, label='저장 표본 %s' % ('A', 'B')[index],
                            path='samples/' + path.name, shape=list(x.shape), sha256=sha(path), bytes=path.stat().st_size,
                            expected=saved['best'][rows].mean(0).tolist(), rows=rows, x=x))
    checks, members = [], []
    tolerance = 1e-4
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sample_predictions = [dict() for _ in samples]
    start = time.monotonic()
    for index, m in enumerate(source['members']):
        assert sha(Path(m['checkpoint'])) == m['checkpoint_sha256']
        family, fold = m['family'], m['fold']
        key = 'bilateral_%d' % fold if family == 'bilateral' else '%s_%d_%s' % (family, fold, m['tag'])
        ck = torch.load(m['checkpoint'], map_location='cpu', weights_only=True)
        model = prior.new_model(family)
        model.load_state_dict(ck['model'], strict=True)
        wrapper = ExportMember(model, family, ck['lo'], ck['span']).eval()
        path = assets / 'models' / (key + '.onnx')
        # The legacy exporter preserves this existing model code, including einops.
        torch.onnx.export(wrapper, (torch.from_numpy(samples[0]['x'][:2]),), str(path),
                          input_names=['features'], output_names=['raw_scores'], opset_version=17,
                          dynamic_axes={'features': {0: 'windows'}, 'raw_scores': {0: 'windows'}},
                          dynamo=False, external_data=False)
        onnx.checker.check_model(str(path))
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        errors = []
        for sample, predictions in zip(samples, sample_predictions):
            x = sample['x']
            reference = prior.forward(model, family, x, ck['lo'], ck['span'])
            actual = session.run(None, {'features': x})[0].astype(np.float64)
            a, b = m['calibration']
            calibrated = actual * a + b
            saved_error = float(np.abs(reference * a + b - saved[key][sample['rows']]).max())
            error = float(np.abs(calibrated - saved[key][sample['rows']]).max())
            assert saved_error <= tolerance, (key, 'source replay', saved_error)
            assert error <= tolerance, (key, 'ONNX', error)
            # Exercise the dynamic dimension at one window as well as thirteen.
            single = session.run(None, {'features': x[:1]})[0].astype(np.float64) * a + b
            single_error = float(np.abs(single - saved[key][sample['rows'][:1]]).max())
            assert single_error <= tolerance, (key, 'single', single_error)
            predictions[key] = calibrated.mean(0)
            errors.append(dict(sample=sample['id'], source_replay_max=saved_error,
                               onnx_max=error, single_window_max=single_error))
        del session
        members.append(dict(id=key, family=family, fold=fold, path='models/' + path.name,
                            sha256=sha(path), bytes=path.stat().st_size, calibration=m['calibration'],
                            source_checkpoint_sha256=m['checkpoint_sha256']))
        checks.append(dict(id=key, samples=errors))
        print('%d/54 %s %.1fs' % (index + 1, key, time.monotonic() - start), flush=True)
    assert len(members) == 54
    selections = [{k: row[k] for k in ('outer', 'cnn_weight', 'fraction')} for row in source['selections']]
    fusion_checks = []
    for sample, predictions in zip(samples, sample_predictions):
        value = combine(predictions, members, selections)
        error = float(np.abs(value - sample['expected']).max())
        assert error <= tolerance, ('fusion', error)
        fusion_checks.append(dict(sample=sample['id'], expected=sample['expected'], onnx=value.tolist(), max_error=error))
        sample.pop('x')
        sample.pop('rows')
    manifest = dict(schema=1, predictor='frozen54-fusion', tolerance=tolerance,
                    source_manifest_sha256=sha(SOURCE), input_layout='windows,side(L/R),time124,features12',
                    feature_order='ankle,knee,hip,big-toe xy; knee/ankle angles; toe-ankle and bilateral ankle distances',
                    preprocessing='Existing frozen full-record preprocessing performed on PC; not part of phone timing.',
                    members=members, selections=selections, samples=samples,
                    runtime=dict(web_version='1.31.1', provider='wasm', threads=1),
                    model_bytes=sum(m['bytes'] for m in members), optimizer_steps=0)
    save(assets / 'manifest.json', manifest)
    save(HERE / 'export-verification.json', dict(passed=True, source_manifest_sha256=sha(SOURCE),
         manifest_sha256=sha(assets / 'manifest.json'), members=len(members), samples=2, checks=checks,
         fusion=fusion_checks, tolerance=tolerance, versions=dict(torch=str(torch.__version__),
         onnx=onnx.__version__, onnxruntime=ort.__version__, numpy=np.__version__),
         smartphone_tested=False, optimizer_steps=0, seconds=time.monotonic() - start))
    print('FROZEN54_EXPORT_AND_PC_REPLAY_PASSED', flush=True)


if __name__ == '__main__':
    main()
