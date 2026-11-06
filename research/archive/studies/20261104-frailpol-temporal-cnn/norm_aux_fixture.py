"""Synthetic fixtures for the paired auxiliary-loss experiment, not accuracy tests."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');sys.path.insert(0,str(H));from norm_aux import Model,preprocess
from native_motion import Model as NativeModel
torch.set_num_threads(2);torch.manual_seed(20261106);reference=NativeModel();torch.manual_seed(20261106);full=Model(24,False);assert all(torch.equal(value,full.core.state_dict()[key]) for key,value in reference.state_dict().items())
torch.manual_seed(20261106);plain=Model(6,False);afterplain=torch.get_rng_state();torch.manual_seed(20261106);aux=Model(6,True);afteraux=torch.get_rng_state();assert torch.equal(afterplain,afteraux);assert all(torch.equal(value,aux.core.state_dict()[key]) for key,value in plain.core.state_dict().items())
rng=np.random.default_rng(20261106);x=torch.from_numpy(rng.normal(size=(8,7,208)).astype(np.float32));x[:,-1]=1;f=torch.from_numpy(rng.normal(size=(8,2)).astype(np.float32));plain.train();aux.train();torch.manual_seed(9191);p=plain(x,f);torch.manual_seed(9191);a,logits=aux.forward_all(x,f);assert torch.equal(p,a)
aux.zero_grad(set_to_none=True);target=torch.from_numpy(rng.integers(0,2,(8,2)).astype(np.float32));torch.nn.functional.binary_cross_entropy_with_logits(logits,target).backward();assert all(param.grad is None for param in aux.core.head.parameters());assert any(param.grad is not None and torch.any(param.grad!=0) for param in aux.core.conv.parameters());assert all(param.grad is not None for param in aux.aux.parameters())
y=np.array([8,9,12,15,20,30.]);d=dict(symnorm=np.ones((6,6,208)),F=np.ones((6,2)),valid=np.ones((6,1,208),bool),y=y);prep=preprocess(d,np.arange(6),'symnorm_aux_tail4');assert np.array_equal(prep['aux_positive_weight'],[.5,2.]);d['y']=np.ones(6)*8
try:preprocess(d,np.arange(6),'symnorm_aux_tail4')
except AssertionError:empty_class_rejected=True
else:raise AssertionError('missingauxclassnotrejected')
record=dict(status='passed_synthetic_implementation_only',full_core_initialization_matches_frozen_native_model_exact=True,norm_plain_vs_aux_core_initialization_exact=True,dropout_rng_and_training_forward_TUG_exact_before_learning=True,auxiliary_loss_updates_encoder_and_aux_head_but_not_main_TUG_head_directly=True,threshold_equalities_and_training_class_weights_verified=True,missing_positive_aux_class_rejected=empty_class_rejected,code_sha256=hashlib.sha256((H/'norm_aux.py').read_bytes()).hexdigest(),limits='Syntheticvaluesonly;notclinicalaccuracyorcompletionof100actualfits/325checkpointreview. Existingtrainingunchanged.');(H/'norm-aux/implementation-verification.json').write_text(json.dumps(record,indent=2));print(record)
