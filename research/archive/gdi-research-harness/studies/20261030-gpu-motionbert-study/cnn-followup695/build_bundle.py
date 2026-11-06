"""Package only authorized development695 inputs; do not open reserved numeric data."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
BUNDLE = HERE / 'bundle'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    source = STUDY / 'expansion695/train_only.npz'
    verified = read(STUDY / 'expansion695/analysis/gpu/verification.json')
    assert sha(source) == verified['dataset_sha256']
    with np.load(source, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in ('x', 'y', 'person_id', 'video_id', 'fold', 'cohort')}
    d['x'] = d['x'][:, 0]
    np.savez_compressed(BUNDLE / 'data.npz', **d)
    cpu = STUDY.parent / '20261030-cpu-head-study/followup'
    cpudata = cpu / 'expanded695-pls-study/development695.npz'
    metadata = read(cpu / 'expanded695-pls64-study/oof.metadata.json')
    predictions = cpu / 'expanded695-pls64-study/oof_predictions.npz'
    assert sha(cpudata) == metadata['dataset_sha256']
    assert sha(predictions) == metadata['predictions_sha256']
    with np.load(cpudata, allow_pickle=False) as z:
        for k in ('person_id', 'video_id', 'fold'):
            assert np.array_equal(d[k], z[k]), k
        # Inspect the established CPU schema rather than infer target order.
        label_key = 'y' if 'y' in z.files else 'Y'
        assert np.array_equal(d['y'], z[label_key])
    with np.load(predictions, allow_pickle=False) as z:
        for k in ('person_id', 'video_id', 'fold'):
            assert np.array_equal(d[k], z[k]), k
        methods = z['methods'].tolist()
        ref = {name: z['predictions'][methods.index(key)].copy()
               for name, key in [('ridge', 'ridge'), ('pls', 'pls64')]}
    with np.load(STUDY / 'expansion695/analysis/gpu/last_stage_oof.npz', allow_pickle=False) as z:
        for k in ('person_id', 'video_id', 'fold'):
            assert np.array_equal(d[k], z[k])
        ref['motionbert_last_stage'] = z['predictions'][:, 0].copy()
    def mae(p):
        e = np.abs(p - d['y']).mean(1)
        return float(np.mean([e[d['person_id'] == k].mean() for k in np.unique(d['person_id'])]))
    references = {k + '_mae': mae(v) for k, v in ref.items()}
    references.update({'people': 695, 'records': 1064, 'same_ids_folds_labels_verified': True,
                       'cpu_predictions_sha256': sha(predictions), 'cpu_data_sha256': sha(cpudata),
                       'reference_only_not_training_input': True})
    save(BUNDLE / 'reference_metrics.json', references)
    np.savez_compressed(BUNDLE / 'reference_predictions.npz', **ref)
    files = ['tools/stanford-sample-check/prepare_ridge_data.py',
             'gdi-research-harness/short_window_contract.py',
             'outputs/motionbert-smoke-20261015/assets/mmpose_inference_3d.py']
    sources = {name: sha(ROOT / name) for name in files}
    joints = ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
              'left_ankle', 'spine', 'thorax', 'nose', 'head_eye_midpoint', 'left_shoulder',
              'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist']
    contract = {'source_dataset_sha256': sha(source), 'data_sha256': sha(BUNDLE / 'data.npz'),
                'source_code_sha256': sources, 'people': 695, 'records': 1064,
                'original_shape': [1064, 1, 124, 17, 3], 'bundle_shape': list(d['x'].shape),
                'cnn_shape': [1064, 51, 124], 'channels': ['x', 'y', 'confidence'], 'joints': joints,
                'targets': ['left_GDI', 'right_GDI'], 'initial_old_people': 175, 'additional_people': 520,
                'missing_confidence_points': int((d['x'][..., 2] == 0).sum()),
                'missing_lower_body_points': int((d['x'][:, :, 1:7, 2] <= 0).sum()),
                'input_channel_ranges': [[float(d['x'][..., k].min()), float(d['x'][..., k].max())] for k in range(3)],
                'normalization': 'per-window valid-joint xy bounding square to [-1,1], clipping retained; confidence unchanged',
                'missingness': 'unchanged prepared input; no new interpolation/padding/smoothing',
                'orientation': 'inherited stored image axes; no added walking-direction flip or side swap',
                'window': 'first124 of eligible longest consecutive lower-body-observed run; not necessarily first video frame',
                'reserved223_and_old37_38_numeric_data_opened': False,
                'teammate_code_available': False,
                'implementation': 'new PyTorch adaptation of official CNN topology, not original-paper reproduction'}
    save(BUNDLE / 'input_contract.json', contract)
    codefiles = list(BUNDLE.glob('*.py'))
    for p in codefiles:
        ast.parse(p.read_text(encoding='utf-8'), feature_version=(3, 8))
    # Ensure scripts copied to Linux have LF line endings.
    for p in list(BUNDLE.glob('*.sh')) + list(BUNDLE.glob('*.slurm')):
        p.write_bytes(p.read_text(encoding='utf-8').replace('\r\n', '\n').encode('utf-8'))
    included = [p for p in sorted(BUNDLE.rglob('*')) if p.is_file()
                and p.name != 'manifest.json' and not any(x in p.parts for x in ('checks', 'runs', 'logs', '__pycache__'))]
    manifest = {p.relative_to(BUNDLE).as_posix(): sha(p) for p in included}
    save(BUNDLE / 'manifest.json', manifest)
    archive = HERE / 'gdi-cnn695.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in included + [BUNDLE / 'manifest.json']:
            z.write(p, arcname='gdi-cnn695-20261030/' + p.relative_to(BUNDLE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in manifest.items():
            assert hashlib.sha256(z.read('gdi-cnn695-20261030/' + name)).hexdigest() == digest
    save(HERE / 'package_verification.json', {'passed': True, 'zip_sha256': sha(archive),
         'bytes': archive.stat().st_size, 'files': len(manifest) + 1,
         'data_identical_to_expansion695': True, 'reference_labels_and_folds_matched': True,
         'python38_syntax_checked': [p.name for p in codefiles], 'reference_metrics': references})
    print(json.dumps({'zip': str(archive), 'bytes': archive.stat().st_size, 'references': references}))


if __name__ == '__main__':
    main()
