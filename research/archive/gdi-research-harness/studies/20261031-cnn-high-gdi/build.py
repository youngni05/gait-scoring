"""Freeze two surgical copies of the completed multiwindow CNN baseline."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'20261030-gpu-motionbert-study/cnn-multiwindow695/bundle'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj): p.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def main():
    provenance = {}
    for mode in ('high90_x3', 'rare_sqrt_cap4'):
        dest = HERE/mode/'bundle'
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('data.npz','model.py','windows.py','input_contract.json','reference_metrics.json'):
            shutil.copy2(BASE/name, dest/name)
        for name in ('weighting.py','audit.py','worker.py'):
            shutil.copy2(HERE/name, dest/name)
        source = (BASE/'run.py').read_text(encoding='utf-8')
        old = '    example_weights=torch.from_numpy(x.side_weights())'
        new = "    weights, weight_audit = make_example_weights(x, y, c['score_weighting'])\n    save(folder / 'weighting.json', weight_audit)\n    example_weights=torch.from_numpy(weights)"
        assert source.count(old) == 1
        source = source.replace('from windows import WindowSet, weighted_mse', 'from windows import WindowSet, weighted_mse\nfrom weighting import example_weights as make_example_weights')
        source = source.replace(old, new)
        source = source.replace("c['stage'] == 'paper_feature_multiwindow_no_search'", "c['stage'] == 'high_gdi_fixed_weight_comparison'")
        source = source.replace("'record_balanced_loss':True", "'record_balanced_loss':True, 'score_weighting':c['score_weighting']")
        (dest/'run.py').write_text(source, encoding='utf-8')
        c = json.loads((BASE/'protocol.json').read_text(encoding='utf-8-sig'))
        c.update(name='cnn_multiwindow_'+mode, stage='high_gdi_fixed_weight_comparison', score_weighting=mode,
                 wall_time_seconds=60000, loss='baseline_record_window_weights_times_train_only_normalized_score_weights_MSE_plus_L2',
                 note='Fixed two-condition development comparison. All other training/selection/calibration settings unchanged. No protected cohorts or external normal labels.')
        save(dest/'protocol.json', c)
        manifest = {p.name:digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='manifest.json'}
        save(dest/'manifest.json', manifest)
        archive = HERE/mode/'bundle.tar.gz'
        with tarfile.open(archive, 'w:gz') as tf:
            for p in sorted(dest.iterdir()): tf.add(p, arcname=p.name)
        with tarfile.open(HERE/mode/'code-only.tar.gz', 'w:gz') as tf:
            for p in sorted(dest.iterdir()):
                if p.name != 'data.npz': tf.add(p, arcname=p.name)
        provenance[mode] = {'baseline_run_sha256':digest(BASE/'run.py'), 'unchanged_data_sha256':digest(dest/'data.npz'),
                            'manifest_sha256':digest(dest/'manifest.json'), 'archive_sha256':digest(archive)}
    save(HERE/'provenance.json', provenance)
    print(json.dumps(provenance))
if __name__ == '__main__': main()
