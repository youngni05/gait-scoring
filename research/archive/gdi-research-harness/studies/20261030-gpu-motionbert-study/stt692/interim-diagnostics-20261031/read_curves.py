"""Read-only remote snapshot; no fitting and no outer evaluation predictions."""
import datetime, hashlib, json, pathlib, subprocess
import numpy as np

result = {'checked_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
          'queue': subprocess.check_output(['squeue','-h','-u','RESEARCH_USER','-o','%i|%T|%L|%R'], text=True),
          'experiments': {}}
for name in ('multiwindow', 'bilateral'):
    root = pathlib.Path('/abr/RESEARCH_USER/gdi-stt692-' + name + '-20261031')
    fits = []
    for path in sorted((root/'runs/main').glob('fold*/*/curve.json')):
        content = path.read_bytes()
        rows = json.loads(content)
        if not rows:
            continue
        selected = {1, 3, 20, rows[-1]['epoch']}
        if 'validation_person_mae' in rows[0]:
            selected.add(min(rows, key=lambda r:r['validation_person_mae'])['epoch'])
        probes = []
        for row in rows:
            if row['epoch'] not in selected:
                continue
            raw_path = path.parent/'raw_predictions'/('epoch%04d.npz' % row['epoch'])
            digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            assert digest == row['raw_predictions_sha256']
            with np.load(raw_path, allow_pickle=False) as z:
                raw = z['train_raw'].astype(float)
            assert np.isfinite(raw).all()
            probes.append({'epoch': row['epoch'], 'raw_sha256': digest,
                           'train_raw_std': float(raw.std()), 'train_raw_range': [float(raw.min()), float(raw.max())]})
        fits.append({'fit': str(path.parent.relative_to(root/'runs/main')),
                     'curve_sha256': hashlib.sha256(content).hexdigest(),
                     'complete': (path.parent/'fit_complete.json').exists(), 'rows': rows, 'probes': probes})
    result['experiments'][name] = {'fits': fits, 'selections': {
        p.parent.name: json.loads(p.read_text()) for p in sorted((root/'runs/main').glob('fold*/selection.json'))}}
print(json.dumps(result, allow_nan=False))
