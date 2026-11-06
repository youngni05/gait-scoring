"""Reuse the verified two-worker transport/gate workflow for this approved comparison."""
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H.parent/'20261101-gdi-cnn-multiseed/manage.py').read_text(encoding='utf-8')
s=s.replace('01a19527-6510-7a70-874d-3e58f91b1383','01a19552-3eb5-7020-9424-1e27c9d7f936')
s=s.replace("AUTOMATION='gdi-cnn'","AUTOMATION='gdi'")
s=s.replace('gdi-cnn-multiseed-20261101','gdi-pairloss-cnn-20261101')
s=s.replace('studies/20261101-gdi-cnn-multiseed','studies/20261101-gdi-pairloss-cnn')
s=s.replace('SEEDS=(20261101,20261102)','SEEDS=(0,1)').replace('[20261101,20261102]','[0,1]')
s=s.replace("name='gdi-seed%d'%SEED","name='gdi-pairloss-%d'%SEED")
s=s.replace("'--time=08:00:00'","'--time=48:00:00'")
s=s.replace("id='cnn-seed%d'%v['seed']","id='cnn-pairloss-arm%d'%v['seed']")
s=s.replace("baselines={'best_blend':6.515816569178735}","baselines={'existing_single_CNN_visit_MAE':6.866655321854305,'existing_single_CNN_change_MAE':None}")
s=s.replace('Use README/manage.py fetch pilot/full, review per seed and compare.py. Verify every epoch, original batch order, inner-only epoch selection, training-only normalization/OLS and five final CPU64 replays. Never weaken 1e-10 coefficient/metric or 1e-4 inference tolerance. Full 50 fits and 10 outer folds plus predeclared three-seed equal average required before final reporting.',
            'See README and protocol: two loss arms 0/1, each25fits5outerfolds. Fetch and review each pilot then local approval gate. Audit all epochs, unchanged state batch order, train-only pair sampling, BN/RNG preservation, inner-only visit-MAE epoch selection, train-only scaling/OLS,5finalCPU64replays per arm. Compare change MAE/direction/LR/first-age-visit/allvisit; no ensemble or weight selection. Tolerances1e-10 metrics/coefficients,1e-4 model inference. Complete both arms before reporting performance.')
s=s.replace("save(H/'main-approval.json',dict(passed=True,pilot_receipt_sha256=sha(H/'checks/pilot-receipt.json'),approved_seeds=list(SEEDS)))","save(H/'main-approval.json',dict(passed=True,pilot_receipt_sha256=sha(H/'checks/pilot-receipt.json'),approved_arms=list(SEEDS)))")
s=s.replace("def deploy():\n", "def deploy():\n    permission=read(H/'transfer-approval.json')\n    assert permission['approved'] and permission['destination']==REMOTE, 'Explicit transfer approval required'\n")
(H/'manage.py').write_text(s,encoding='utf-8',newline='\n')
print('Prepared manager for this thread; seed0/seed1 denote loss arms, not different initialization.')
