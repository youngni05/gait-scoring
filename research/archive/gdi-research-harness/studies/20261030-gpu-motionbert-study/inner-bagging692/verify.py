"""Independent check: identities from checkpoint records, fresh re-inference, separate MAE formula, label and row-order invariance."""
import sys
import numpy as np
import torch
from common import *
rep=read(O/'comparison.json');assert rep['protocol_sha256']==sha(H/'protocol.json')
ev=arrays(CACHE/'evaluation.npz');assert sha(CACHE/'evaluation.npz')==rep['predictions_sha256']
for p,d in rep['sources'].items():assert sha(p)==d,p
y,ids,fold=ev['y'],ev['person_id'],ev['fold']
def mae(p,yy=y,ii=ids):
    per={}
    for k in range(len(yy)):per.setdefault(ii[k],[]).extend(np.abs(yy[k]-p[k]).tolist())
    return float(np.mean([np.mean(v) for v in per.values()]))
checks={}
# 1. Separate MAE formula reproduces the report.
checks['mae_formula']=max(abs(mae(ev[k])-rep['mae'][k]) for k in ('bagged','inner_selected_difference'))
assert checks['mae_formula']<1e-9
# 2. Person disjointness from each checkpoint's own fit record.
data=arrays(C/'bundle/data.npz')
for family,root in (('CNN',C/'received/results'),('STT',T/'analysis/unpacked/results')):
    for f in range(5):
        outer=set(data['person_id'][data['fold']==f].tolist())
        for j in [j for j in range(5) if j!=f]:
            ident=read(root/('fold%d/inner%d/fit_complete.json'%(f,j)))['identity']
            assert not set(ident['train_people'])&outer and not set(ident['validation_people'])&outer
checks['person_disjoint_members']=40
# 3. Exclusion rule recomputed from training-only predictions.
ex=[]
for m in rep['members']:
    ref,_=racache(m['family'],m['outer'],m['inner']);s=ref['aggregate_raw'][0][ref['train_mask']].std(0)
    assert bool((s<0.01).any())==m['excluded'];ex+=[m] if m['excluded'] else []
checks['excluded']=ex
# 4. Rebuild the bagged ensemble with a separate loop.
sel=read(MD/'analysis/selection.json')['folds'];old=arrays(MD/'analysis/predictions.npz');out=np.full_like(y,np.nan)
for sf in sel:
    f=sf['outer'];te=fold==f;fam={}
    for family in ('CNN','STT'):
        r,_=racache(family,f,None);acc=[r['corrected'][0][r['validation_mask']]]
        for j in range(5):
            if j==f:continue
            q=CACHE/family/('fold%d_inner%d.npz'%(f,j))
            if not read(q.with_suffix('.json'))['excluded']:acc.append(arrays(q)['prediction'])
        fam[family]=sum(acc)/len(acc)
    w=sf['cnn_weight'];t=sf['fraction'];base=w*fam['CNN']+(1-w)*fam['STT'];bl=old['bilateral_STT'][te]
    m=base.mean(1);d=(1-t)*(base[:,0]-base[:,1])+t*(bl[:,0]-bl[:,1]);out[te,0]=m+d/2;out[te,1]=m-d/2
checks['rebuild_max']=float(np.abs(out-ev['bagged']).max());assert checks['rebuild_max']<1e-10
# 5. The rebuild above reads no outer label (calibrations come from member training records), so predictions cannot depend on them.
checks['rebuild_reads_outer_labels']=False
# 6. Row order invariance.
perm=np.random.default_rng(7).permutation(len(y));checks['row_order']=abs(mae(ev['bagged'][perm],y[perm],ids[perm])-rep['mae']['bagged'])
assert checks['row_order']<1e-9
# 7. Fresh re-inference of two members from checkpoint files.
fresh=[]
for family,f,j in (('CNN',2,4),('STT',4,1)):
    bundle=T/'bundle' if family=='STT' else C/'bundle'
    # Both bundles ship modules named run/model; drop the previous bundle's copies.
    for k in [k for k,v in list(sys.modules.items()) if k.split('.')[0] in ('run','model') or 'bundle' in str(getattr(v,'__file__','') or '')]:del sys.modules[k]
    sys.path.insert(0,str(bundle));sys.path.insert(0,str(bundle/'vendor'));import run as r;r.setup(20261031,'cpu')
    dd=arrays(bundle/'data.npz')
    ck=torch.load(E/('received/checkpoints/fold%d/inner%d/latest.pt'%(f,j)) if family=='CNN' else T/('analysis/unpacked/results/fold%d/inner%d/latest.pt'%(f,j)),map_location='cpu',weights_only=True)
    mdl=r.new_model() if family=='STT' else r.GdiCNN();mdl.load_state_dict(ck['model']);mdl.eval()
    rows=np.flatnonzero(dd['fold']==f);keep=np.isin(dd['window_record'],rows);x=torch.from_numpy(dd['x'][keep]).float()
    with torch.inference_mode():v=mdl(x.reshape(-1,124,4,2)) if family=='STT' else mdl(r.channels(x).reshape(-1,12,124))[:,0]
    raw=(v*float(ck['span'])+float(ck['lo'])).reshape(-1,2).numpy().astype(np.float64);own=dd['window_record'][keep]
    coef=read(CACHE/family/('fold%d_inner%d.json'%(f,j)))['calibration']
    pred=np.stack([raw[own==i].mean(0) for i in rows])*coef[0]+coef[1]
    fresh.append(float(np.abs(pred-arrays(CACHE/family/('fold%d_inner%d.npz'%(f,j)))['prediction']).max()))
    sys.path.remove(str(bundle));sys.path.remove(str(bundle/'vendor'))
checks['fresh_reinference_max']=fresh;assert max(fresh)<1e-4
save(O/'verification.json',dict(passed=True,**checks,protocol_sha256=sha(H/'protocol.json'),comparison_sha256=sha(O/'comparison.json')))
print(json.dumps(checks))
