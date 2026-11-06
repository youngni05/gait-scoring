"""Export a verified CPU artifact declaration and analyze it under A's fixed rules."""
from pathlib import Path
import evaluate as e

d, baseline, hashes = e.data()
path = e.HERE.parent / 'runs/main/all_predictions.npz'
model_files = [e.HERE.parent / f'runs/main/head_only_seed20261028_fold{k}/checkpoint.pt' for k in range(5)]
meta = e.make_metadata(path, d, hashes, model_files, 'cpu_head_only')
e.write_json(e.HERE / 'cpu_predictions.metadata.json', meta)
e.evaluate(path, e.HERE / 'cpu_predictions.metadata.json', e.HERE / 'cpu_comparison', 'cpu_head_only')
