"""Save raw outputs of every mixed member on external sets. No fitting."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=HERE.parents[2]
BIG=Path('D:/gait-research-data/D_P/unseen-normal-test-20261105');SRC=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
fam=sys.argv[1];B=ROOT/'studies/20261104-ljmu100-mixed/bundle'/fam;sys.path[:0]=[str(B),str(B/'vendor')]
spec=importlib.util.spec_from_file_location('rr',B/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
r.setup(20261028,'cpu');r.torch.set_num_threads(8)
c=read(B/'protocol.json');meta=read(PARENT/'preparation.json');assert sha(BIG/'inputs.npz')==meta['inputs_sha256']
Z=np.load(BIG/'inputs.npz');names=['live','self','ljmu_original'];data={}
for n in names:
    x=Z[n+'__x']
    if fam!='CNN':x=np.ascontiguousarray(x[...,:8].reshape(-1,2,124,4,2))
    data[n]=r.WindowSet(x,Z[n+'__owner'],int(Z[n+'__n']))
fold=np.array(meta['ljmu_fold']);out={};checked=0
for f in range(5):
    root=SRC/'received/data/runs/mixed'/fam/f'fold{f}';done=read(root/'complete.json')
    for m in done['members']:
        path=root/m['tag']/'latest.pt';assert sha(path)==m['checkpoint_sha256'];checked+=1
        ck=r.torch.load(path,map_location='cpu',weights_only=True)
        model=r.GdiCNN(c['dropout']) if fam=='CNN' else r.new_model();model.load_state_dict(ck['model'],strict=True)
        for n in names:
            x=data[n] if n!='ljmu_original' else data[n][np.flatnonzero(fold==f)]
            out[f"{n}__{f}__{m['tag']}"]=r.predict(model,x,ck['lo'],ck['span'],'cpu')
    print(fam,f,flush=True)
np.savez_compressed(BIG/f'member-raw-{fam}.npz',**out)
(HERE/f'infer-{fam}.json').write_text(json.dumps(dict(family=fam,checkpoints_hash_checked=checked,prediction_sha256=sha(BIG/f'member-raw-{fam}.npz')),indent=1),encoding='utf-8')
print('done',fam,checked,flush=True)
