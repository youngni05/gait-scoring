"""External-set inference for matched arms. Usage: infer.py <arm> <family>. No fitting."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=HERE.parents[2]
BIG=Path('D:/gait-research-data/D_P/unseen-normal-test-20261105');SRC=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104');MATCH=Path('D:/gait-research-data/D_P/matched-normal100-20261105')
OLD=ROOT/'studies/20261104-ljmu100-mixed/bundle'
arm,fam=sys.argv[1:3];B=OLD/fam;sys.path[:0]=[str(B),str(B/'vendor')]
spec=importlib.util.spec_from_file_location('rr',B/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
r.setup(20261028,'cpu');r.torch.set_num_threads(8)
c=read(B/'protocol.json');meta=read(PARENT/'preparation.json');assert sha(BIG/'inputs.npz')==meta['inputs_sha256']
Z=np.load(BIG/'inputs.npz');data={}
def wrap(x,o,n):
    if fam!='CNN':x=np.ascontiguousarray(x[...,:8].reshape(-1,2,124,4,2))
    return r.WindowSet(x,o,n)
for n in ['live','self','ljmu_original']:data[n]=wrap(Z[n+'__x'],Z[n+'__owner'],int(Z[n+'__n']))
replay=None
if arm!='baseline':
    with np.load(MATCH/arm/'data.npz') as z:
        add=np.flatnonzero(z['source']!='Stanford');keep=np.isin(z['window_record'],add);m={int(k):i for i,k in enumerate(add)}
        data['added']=wrap(z['x'][keep],np.array([m[int(k)] for k in z['window_record'][keep]]),len(add))
base=read(OLD/'baseline-models.json')
def members(f):
    if arm=='baseline':
        return [(SRC/m['checkpoint'],m['checkpoint_sha256'],m['tag'],np.array(m['calibration']),False,None) for m in base if m['family']==fam and m['fold']==f]
    root=MATCH/'received/data'/arm/'runs/mixed'/fam/f'fold{f}';done=read(root/'complete.json')
    return [(root/m['tag']/'latest.pt',m['checkpoint_sha256'],m['tag'],np.array(m['calibration']),m['excluded_from_bag'],root/m['tag']/'replay.npz') for m in done['members']]
out={};checked=0;maxreplay=0.
names=[n for n in data if n!='added']
for f in range(5):
    bag={n:[] for n in names};refit={}
    for path,digest,tag,coef,excluded,rep in members(f):
        assert sha(path)==digest,path;checked+=1
        ck=r.torch.load(path,map_location='cpu',weights_only=True)
        model=r.GdiCNN(c['dropout']) if fam=='CNN' else r.new_model();model.load_state_dict(ck['model'],strict=True)
        for n in names:
            p=r.predict(model,data[n],ck['lo'],ck['span'],'cpu')*coef[0]+coef[1]
            if not excluded:bag[n].append(p)
            if tag=='refit':refit[n]=p
        if rep is not None:
            with np.load(rep) as z:
                pos={int(v):i for i,v in enumerate(z['train_row'])};saved=z['train_raw'][[pos[int(v)] for v in add]]
            maxreplay=max(maxreplay,float(np.abs(r.predict(model,data['added'],ck['lo'],ck['span'],'cpu')-saved).max()))
    for n in names:
        out[f'{n}__{f}__bag']=np.mean(bag[n],axis=0);out[f'{n}__{f}__refit']=refit[n]
    print(arm,fam,f,flush=True)
np.savez_compressed(BIG/f'dose-{arm}-{fam}.npz',**out)
(HERE/f'infer-{arm}-{fam}.json').write_text(json.dumps(dict(arm=arm,family=fam,checkpoints_hash_checked=checked,replay_max_error=maxreplay,prediction_sha256=sha(BIG/f'dose-{arm}-{fam}.npz')),indent=1),encoding='utf-8')
print('done',arm,fam,checked,maxreplay,flush=True)
