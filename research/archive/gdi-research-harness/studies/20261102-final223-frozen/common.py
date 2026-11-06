from pathlib import Path
import hashlib,json,importlib.util,sys
HERE=Path(__file__).resolve().parent
H=HERE.parents[1];ROOT=H.parent
STUDY=H/'studies/20261030-gpu-motionbert-study'
BIG=Path('D:/gait-research-data/D_P/final223-frozen-20261102')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(2**20),b''):h.update(v)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
prior=module('prior_frozen',H/'studies/20261101-self-video-gdi/predict.py')
np=prior.np;torch=prior.torch
torch.set_num_threads(4);torch.manual_seed(20261102);torch.use_deterministic_algorithms(True)
def locked():
    f=read(HERE/'freeze.json')
    assert sha(HERE/'protocol.json')==f['protocol_sha256']
    assert sha(HERE/'models.json')==f['models_sha256']
    if (HERE/'code-seal.json').exists():
        for name,digest in read(HERE/'code-seal.json').items():assert sha(HERE/name)==digest,(name,'code changed')
    return read(HERE/'protocol.json')
def aggregate(values,owners,n):
    sums=np.zeros((n,2),dtype=np.float64);np.add.at(sums,owners,values)
    return sums/np.bincount(owners,minlength=n)[:,None]
def person_errors(y,p,ids):
    return np.array([np.abs(y[ids==i]-p[ids==i]).mean() for i in np.unique(ids)])
def combine(cnn,stt,bilateral,w,t):
    base=w*cnn+(1-w)*stt;m=base.mean(1)
    d=(1-t)*(base[:,0]-base[:,1])+t*(bilateral[:,0]-bilateral[:,1])
    return np.column_stack([m+d/2,m-d/2])
