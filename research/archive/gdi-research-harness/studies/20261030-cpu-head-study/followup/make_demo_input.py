"""Create one deterministic anonymous cached example; no choosing a favorable result."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import cpu_head as m
import numpy as np

d, heads, baseline, fingerprints = m.source_data()
index = 0  # Fixed first record, not selected by prediction error.
fold = int(d['fold'][index])
out = HERE / 'demo'
out.mkdir(exist_ok=True)
np.savez_compressed(out / 'features.npz', features=d['features'][index])
checkpoint = HERE.parent / f'runs/main/head_only_seed20261028_fold{fold}/checkpoint.pt'
m.save_json(out / 'input_manifest.json', {'selection': 'fixed first record in existing development artifact; not chosen by score',
    'record_index_for_internal_replay': index, 'evaluation_fold': fold,
    'features_sha256': m.digest(out / 'features.npz'), 'checkpoint_sha256': m.digest(checkpoint),
    'checkpoint_path': checkpoint.as_posix(), 'dataset_sha256': fingerprints['train_only.npz'],
    'no_person_or_video_identifier_in_user_facing_demo': True})
print('Created fixed anonymous cached example; evaluation fold:', fold)
