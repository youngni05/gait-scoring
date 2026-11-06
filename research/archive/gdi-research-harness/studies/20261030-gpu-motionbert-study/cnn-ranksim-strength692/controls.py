"""Bind reused controls to the original immutable experiment, without retraining."""
from pathlib import Path
import os,ast
def verify_reuse(r):
 root=Path(os.environ.get('CONTROL_ROOT',r.config()['reuse_control_remote']))
 assert r.sha(root/'manifest.json')==r.config()['reuse_control_manifest_sha256']
 for n,digest in r.load(root/'manifest.json').items():assert r.sha(root/n)==digest
 provenance=r.load(r.ROOT/'source-provenance.json')
 for n,digest in provenance['unchanged_files'].items():assert r.sha(root/n)==r.sha(r.ROOT/n)==digest
 def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
 a,b=funcs(root/'run.py'),funcs(r.ROOT/'run.py')
 for n in provenance['unchanged_functions']:assert a[n]==b[n],n
 old=r.load(root/'protocol.json');new=r.config()
 for k in ['seed','max_epochs','learning_rate','lr_drop_every','lr_drop_factor','dropout','kernel_l2','optimizer','rmsprop_alpha','rmsprop_eps','target_scaling','selection','input','outer_folds','batch_size','data_sha256','rank_interpolation']:assert old[k]==new[k],k
 for variant in ['mse','ranksim']:
  summary=r.load(root/'runs/main'/variant/'summary.json');assert summary['fits']==25
  assert r.sha(root/'runs/main'/variant/'oof_predictions.npz')==summary['oof_sha256']
 return {'passed':True,'root':str(root),'same_training_functions':True,'same_initial_seed_sampler_model_optimizer_and_scaling':True,'new_control_training':False}
