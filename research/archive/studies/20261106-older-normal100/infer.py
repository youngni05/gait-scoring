"""Raw per-window outputs of baseline and mixed members on the older cohort. No fitting."""
import hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tmp/ljmu-cnn-python-deps'))
import numpy as np
BIG=Path('D:/gait-research-data/D_P/older-normal100-20261106');SRC=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
INP=Path('D:/gait-research-data/D_P/frozen-transfer-and-deviation-20261103')
OLD=ROOT/'studies/20261104-ljmu100-mixed/bundle'
fam=sys.argv[1];B=OLD/fam;sys.path[:0]=[str(B),str(B/'vendor')]
spec=importlib.util.spec_from_file_location('rr',B/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
r.setup(20261028,'cpu');r.torch.set_num_threads(4)
c=read(B/'protocol.json')
with np.load(INP/'features.npz') as z:x=z['x']
n=len(x)
if fam!='CNN':x=np.ascontiguousarray(x[...,:8].reshape(-1,2,124,4,2))
data=r.WindowSet(x,np.arange(n),n)
base=read(OLD/'baseline-models.json');out={};checked=0
def run(path,digest):
    global checked
    assert sha(path)==digest,path;checked+=1
    ck=r.torch.load(path,map_location='cpu',weights_only=True)
    model=r.GdiCNN(c['dropout']) if fam=='CNN' else r.new_model();model.load_state_dict(ck['model'],strict=True)
    return r.predict(model,data,ck['lo'],ck['span'],'cpu')
for f in range(5):
    for m in [q for q in base if q['family']==fam and q['fold']==f]:
        out[f"baseline__{f}__{m['tag']}"]=run(SRC/m['checkpoint'],m['checkpoint_sha256'])
    root=SRC/'received/data/runs/mixed'/fam/f'fold{f}';done=read(root/'complete.json')
    for m in done['members']:
        out[f"mixed__{f}__{m['tag']}"]=run(root/m['tag']/'latest.pt',m['checkpoint_sha256'])
    print(fam,f,flush=True)
np.savez_compressed(BIG/f'raw-{fam}.npz',**out)
(HERE/f'infer-{fam}.json').write_text(json.dumps(dict(family=fam,checkpoints_hash_checked=checked,windows=n,prediction_sha256=sha(BIG/f'raw-{fam}.npz'),torch=str(r.torch.__version__)),indent=1),encoding='utf-8')
print('done',fam,checked,flush=True)
