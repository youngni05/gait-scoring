"""ROCKET-inspired fixed multichannel random convolution descriptors, not paper reproduction."""
from pathlib import Path
import sys,json,zipfile,hashlib,os,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');SRC=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103')
sys.path.insert(0,str(H));import features2 as f2
import numpy as np
from signal_pilot import filemap,rawread,save
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sequence(raw,window):
    lo,hi=window['core_start'],window['core_end'];channels=[];ref=raw['sacrum back'].PacketCounter.iloc[lo:hi].to_numpy()
    for sensor in ['left ankle','right ankle','sacrum back']:
        d=raw[sensor];assert d.PacketCounter.is_unique;d=d.set_index('PacketCounter').loc[ref]
        for kind in ['Acc','Gyr']:
            x=d[[kind+'_'+a for a in ['X','Y','Z']]].to_numpy(float)
            if kind=='Gyr':x=np.rad2deg(x)
            x=np.column_stack([x,np.linalg.norm(x,axis=1)])
            for v in x.T:
                v=np.interp(np.linspace(0,1,128),np.linspace(0,1,len(v)),v)
                channels.append((v-v.mean())/max(v.std(),1e-8))
    a=np.array(channels);assert a.shape==(24,128) and np.isfinite(a).all();return a
def kernels():
    rng=np.random.default_rng(20261104);out=[]
    for _ in range(1000):
        length=int(rng.choice([7,9,11]));dilation=int(2**rng.uniform(0,np.log2(127/(length-1))));count=int(rng.integers(1,6));cols=rng.choice(24,count,replace=False);w=rng.normal(size=(count,length));w-=w.mean(axis=1,keepdims=True)
        out.append(dict(channels=cols,weights=w,bias=float(rng.uniform(-1,1)),dilation=dilation,padding=(length-1)*dilation//2 if rng.integers(2) else 0))
    return out
def transform(X,bank,reference=False):
    # float64 deterministic inference; data-independent bank and per-window normalization.
    out=np.zeros((len(X),len(bank)*2),dtype=float)
    for j,k in enumerate(bank):
        a=X[:,k['channels'],:];pad=k['padding'];a=np.pad(a,((0,0),(0,0),(pad,pad)));length=k['weights'].shape[1];d=k['dilation'];n=a.shape[-1]-(length-1)*d
        if reference:
            conv=np.zeros((len(X),n))+k['bias']
            for c in range(len(k['channels'])):
                for t in range(length):conv+=a[:,c,t*d:t*d+n]*k['weights'][c,t]
        else:
            windows=np.lib.stride_tricks.sliding_window_view(a,(length-1)*d+1,axis=2)[:,:,:,::d]
            conv=np.einsum('nctl,cl->nt',windows,k['weights'],optimize=False)+k['bias']
        out[:,2*j]=conv.max(axis=1);out[:,2*j+1]=(conv>0).mean(axis=1)
    return out
def main():
    h=H/'round04';b=B/'round04';h.mkdir(exist_ok=True);b.mkdir(exist_ok=True)
    proto=dict(code_sha256=sha(Path(__file__)),raw_sha256=sha(SRC/'raw/IMU_signals.zip'),single_source_sha256=sha(SRC/'cohorts.json'),multi_source_sha256=sha(B/'round03/features.json'),design='Fixed seed20261104,1000 multichannel random kernels(length7/9/11,log-uniform dilation,random zero padding,Gaussian mean-centered weights,bias[-1,1],1..5random channels), max+positive fraction ->2000features. 3validated sensors * (acc/gyr)*(XYZ+norm)=24channels; interpolate selected core to128samples then per-window channel zscore. Keep original19features to preserve speed, duration and magnitude. Compare original single cycle and previously frozen multi-cycle windows. No global duration/location/clinical variables/labels used. ROCKET-inspired custom multichannel regression experiment, not exact paper implementation.',reference='https://arxiv.org/abs/1910.13051',verification='Read all594 raw sequences twice exact; outside-core perturbation first12; convolution independent explicit multiply/add implementation for first12 all1000filters; full transform deterministic replay. Bank and feature artifacts saved and hashed.',expected_additional_GB=.10)
    if (h/'feature-protocol.json').exists():assert json.loads((h/'feature-protocol.json').read_text())==proto
    else:save(h/'feature-protocol.json',proto)
    rows=[r for r in json.loads((SRC/'cohorts.json').read_text()) if r['status']=='ok' and r['tug'] is not None];assert len(rows)==594
    multi={r['id']:r for r in json.loads((B/'round03/features.json').read_text())};bank=kernels();bank_serial=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in q.items()} for q in bank];save(b/'kernel-bank.json',bank_serial)
    result={};audit={};start=time.monotonic()
    with zipfile.ZipFile(SRC/'raw/IMU_signals.zip') as z:
      mapping=filemap(z)
      for mode in ['single','multi']:
        seq=[];base=[]
        for n,r in enumerate(rows):
            w=r['window'] if mode=='single' else multi[r['id']]['window'];raw=rawread(z,mapping,r['id']);a=sequence(raw,w);again=sequence(rawread(z,mapping,r['id']),w);assert np.array_equal(a,again)
            if n<12:
                ref=raw['sacrum back'].PacketCounter.iloc[w['core_start']:w['core_end']].to_numpy();alter={s:d.copy() for s,d in raw.items()}
                for d in alter.values():d.loc[~d.PacketCounter.isin(ref),d.columns!='PacketCounter']=99999.
                assert np.array_equal(a,sequence(alter,w))
            seq.append(a);base.append([r['features'][k] if mode=='single' else multi[r['id']]['features'][k] for k in r['features']])
        X=np.array(seq);f=transform(X,bank);again=transform(X,bank);assert np.array_equal(f,again)
        ref=transform(X[:12],bank,reference=True);delta=float(np.max(np.abs(ref-f[:12])));assert np.allclose(ref,f[:12],atol=1e-10,rtol=1e-10)
        np.savez_compressed(b/(mode+'-signals.npz'),X=X,ids=np.array([r['id'] for r in rows]))
        result[mode]=np.column_stack([base,f]);audit[mode]=dict(people=len(X),features=result[mode].shape[1],independent_convolution_max_delta=delta);print(mode,audit[mode],round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(b/'features.npz',ids=np.array([r['id'] for r in rows]),**result)
    save(h/'feature-verification.json',dict(status='passed',checks=audit,features_sha256=sha(b/'features.npz'),kernel_bank_sha256=sha(b/'kernel-bank.json'),seconds=time.monotonic()-start))
if __name__=='__main__':main()
