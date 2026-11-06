from pathlib import Path
import os, sys, json, hashlib
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
ROOT=Path('D:/Projects/D_P')
HERE=ROOT/'studies/20261103-sppb-lightgbm'
BIG=Path('D:/gait-research-data/D_P/sppb-lightgbm-20261103')
OLD=ROOT/'studies/20261103-older44-coordination-sppb'
LARGE=ROOT/'studies/20261103-larger-older-sppb'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);e=p-y;low=y<=8;pred=p<=8
    out=dict(n=len(y),MAE=float(abs(e).mean()),RMSE=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),R2=float(1-(e@e)/np.sum((y-y.mean())**2)),spearman=float(spearmanr(y,p).statistic) if np.ptp(p)>0 else None,within1=float((abs(e)<=1).mean()),errors_gt2=int((abs(e)>2).sum()),TP=int((low&pred).sum()),FN=int((low&~pred).sum()),TN=int((~low&~pred).sum()),FP=int((~low&pred).sum()),bands={})
    for name,mask in [('0-4',y<=4),('5-8',(y>=5)&(y<=8)),('9-10',(y>=9)&(y<=10)),('11-12',y>=11)]:
        out['bands'][name]=dict(n=int(mask.sum()),MAE=float(abs(e[mask]).mean()) if mask.any() else None,bias=float(e[mask].mean()) if mask.any() else None)
    return out
def contrasts(y,preds):
    idx=np.random.RandomState(20261103).randint(0,len(y),(10000,len(y)));out={}
    for a,b in [('features_LGBM','speed_LGBM'),('features_GDI_LGBM','features_LGBM'),('features_GDI_LGBM','speed_LGBM'),('features_LGBM','speed_Ridge'),('features_GDI_LGBM','speed_Ridge'),('speed_LGBM','speed_Ridge')]:
        if a not in preds: continue
        d=abs(preds[a]-y)-abs(preds[b]-y)
        out[a+'-'+b]=dict(MAE_difference=float(d.mean()),CI95=np.quantile(d[idx].mean(1),[.025,.975]).tolist())
    return out
