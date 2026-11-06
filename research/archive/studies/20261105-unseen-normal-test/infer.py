"""Fixed checkpoint inference for one model family. No fitting."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BIG=Path('D:/gait-research-data/D_P/unseen-normal-test-20261105');SRC=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
OLD=ROOT/'studies/20261104-ljmu100-mixed/bundle'
fam=sys.argv[1];B=OLD/fam;sys.path[:0]=[str(B),str(B/'vendor')]
spec=importlib.util.spec_from_file_location('rr',B/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
r.setup(20261028,'cpu');r.torch.set_num_threads(8)
c=read(B/'protocol.json');meta=read(HERE/'preparation.json');assert sha(BIG/'inputs.npz')==meta['inputs_sha256']
Z=np.load(BIG/'inputs.npz');names=sorted({k.split('__')[0] for k in Z.files})
data={}
for n in names:
    x=Z[n+'__x'];o=Z[n+'__owner'];m=int(Z[n+'__n'])
    if fam!='CNN':x=np.ascontiguousarray(x[...,:8].reshape(-1,2,124,4,2))
    data[n]=r.WindowSet(x,o,m)
fold=np.array(meta['ljmu_fold'])
base=read(OLD/'baseline-models.json')
def members(arm,f):
    if arm=='baseline':
        return [(SRC/m['checkpoint'],m['checkpoint_sha256'],m['tag'],np.array(m['calibration']),False) for m in base if m['family']==fam and m['fold']==f]
    root=SRC/'received/data/runs/mixed'/fam/f'fold{f}';done=read(root/'complete.json')
    return [(root/m['tag']/'latest.pt',m['checkpoint_sha256'],m['tag'],np.array(m['calibration']),m['excluded_from_bag']) for m in done['members']]
out={};checked=0
for arm in ['baseline','mixed']:
    for f in range(5):
        bag={n:[] for n in names};refit={}
        for path,digest,tag,coef,excluded in members(arm,f):
            assert sha(path)==digest,path;checked+=1
            ck=r.torch.load(path,map_location='cpu',weights_only=True)
            model=r.GdiCNN(c['dropout']) if fam=='CNN' else r.new_model();model.load_state_dict(ck['model'],strict=True)
            for n in names:
                x=data[n] if not n.startswith('ljmu') else data[n][np.flatnonzero(fold==f)]
                p=r.predict(model,x,ck['lo'],ck['span'],'cpu')*coef[0]+coef[1]
                if not excluded:bag[n].append(p)
                if tag=='refit':refit[n]=p
        for n in names:
            out[f'{n}__{arm}__{f}__bag']=np.mean(bag[n],axis=0);out[f'{n}__{arm}__{f}__refit']=refit[n]
        print(fam,arm,f,flush=True)
np.savez_compressed(BIG/f'pred-{fam}.npz',**out)
(HERE/f'infer-{fam}.json').write_text(json.dumps(dict(family=fam,checkpoints_hash_checked=checked,inputs_sha256=meta['inputs_sha256'],prediction_sha256=sha(BIG/f'pred-{fam}.npz'),torch=str(r.torch.__version__)),indent=1),encoding='utf-8')
print('done',fam,checked,flush=True)
