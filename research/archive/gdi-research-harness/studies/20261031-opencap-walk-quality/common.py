"""Signal-consistency controls, not ground-truth tracking-quality labels."""
from pathlib import Path
import sys,json,hashlib,importlib.util
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PRIOR=HERE.parent/'20261031-opencap-walk-pattern'
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
spec=importlib.util.spec_from_file_location('prior_walk_features',PRIOR/'common.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
SEEDS=prior.SEEDS
VIEWS=['speed','quality','pattern','robust']
PATTERN_NAMES=prior.PATTERN_NAMES
VARIANTS=[(w,p) for w in [3,5,7] for p in [6,8,10]]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def load_data(name):return dict(np.load(HERE/name/'input.npz',allow_pickle=False))

def patterns(x,width=5,prominence=8):
    # Same definitions as the prior experiment; only smoothing and prominence vary.
    x=np.asarray(x,dtype=float);n=x.shape[1]
    bad=np.any(np.abs(np.diff(x[:17],axis=1))*30>1000,axis=0)
    bad=np.r_[bad,False]|np.r_[False,bad]
    bad=np.convolve(bad.astype(int),np.ones(7),mode='same')>0
    smooth=np.stack([np.convolve(np.pad(a,(width//2,width//2),mode='edge'),np.ones(width)/width,mode='valid') for a in x[:17]])
    intervals=[];spans=[]
    for knee in [5,11]:
        peaks,_=find_peaks(smooth[knee],distance=12,prominence=prominence)
        peaks=peaks[(peaks>=2)&(peaks<n-2)]
        valid=[(int(a),int(b)) for a,b in zip(peaks[:-1],peaks[1:]) if .6<=(b-a)/30<=2.5 and not bad[a:b+1].any()]
        spans.append(valid);intervals.append(np.array([(b-a)/30 for a,b in valid]))
    out=np.full(14,np.nan);both=all(len(a)>=1 for a in intervals)
    if both:
        duration=np.concatenate(intervals);out[0]=np.median(duration)
        if all(len(a)>=2 for a in intervals):out[1]=duration.std()/duration.mean()
        med=[np.median(a) for a in intervals];out[2]=abs(med[0]-med[1])/np.mean(med)
        for j,cs in enumerate([[5,11],[2,8],[6,12]]):
            rom=[np.median([np.ptp(smooth[c,a:b+1]) for a,b in spans[s]]) for s,c in enumerate(cs)]
            out[3+2*j]=np.mean(rom)
            if np.mean(rom)>1:out[4+2*j]=abs(rom[0]-rom[1])/np.mean(rom)
        corrs=[]
        for side,(hip,knee) in enumerate([(2,5),(8,11)]):
            for a,b in spans[side]:
                h,k=smooth[hip,a:b+1],smooth[knee,a:b+1]
                if h.std()>1 and k.std()>1:corrs.append(np.corrcoef(h,k)[0,1])
        if corrs:out[9]=np.mean(corrs)
    if (~bad).mean()>=.8:
        for j,c in enumerate([1,14,15,16]):out[10+j]=np.diff(np.percentile(smooth[c,~bad],[5,95]))[0]
    return out,{'bad_fraction':float(bad.mean()),'right_cycles':len(spans[0]),'left_cycles':len(spans[1])}

def extract(x):
    original,q=patterns(x)
    variations=np.array([patterns(x,w,p)[0] for w,p in VARIANTS])
    robust=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in variations.T])
    cycles=variations[:,0];available=np.isfinite(cycles);fraction=float(available.mean())
    spread=float(np.ptp(cycles[available])) if available.any() else 0.
    quality=np.r_[q['bad_fraction'],fraction,spread,~np.isfinite(original),~np.isfinite(robust)].astype(float)
    stable=(q['bad_fraction']<=.05 and min(q['right_cycles'],q['left_cycles'])>=2 and available.sum()>=7
            and spread<=.15 and np.isfinite(original).all() and np.isfinite(robust).all())
    return original,robust,quality,stable,dict(q,valid_variants=int(available.sum()),cycle_seconds_spread=spread,stable=bool(stable))

def make_model(family,setting,seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    model=(RandomForestRegressor(n_estimators=128,min_samples_leaf=int(setting),random_state=seed,n_jobs=1,max_features=1.)
           if family=='RF' else Ridge(alpha=float(setting)))
    # The same explicit missingness flags are in B/C/D. Avoid extra copies of them.
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=False,keep_empty_features=True),StandardScaler(),model)
