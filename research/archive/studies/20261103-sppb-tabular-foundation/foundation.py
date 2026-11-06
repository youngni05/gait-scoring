from pathlib import Path
import os, sys, json, hashlib
HERE=Path('D:/Projects/D_P/studies/20261103-sppb-tabular-foundation')
BIG=Path('D:/gait-research-data/D_P/sppb-tabular-foundation-20261103')
for key,value in {'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HOME':str(BIG/'cache/hf'),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1','TABPFN_ALLOW_CPU_LARGE_DATASET':'1','SKB_DATA_DIRECTORY':str(BIG/'cache/skrub'),'MPLCONFIGDIR':str(BIG/'cache/mpl'),'TABPFN_MODEL_CACHE_DIR':str(BIG/'weights/TabPFN-v2-reg')}.items():os.environ[key]=value
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def factory(kind):
    if kind=='TabICLv2':return TabICLRegressor(n_estimators=8,model_path=BIG/'weights/TabICL/tabicl-regressor-v2-20260212.ckpt',allow_auto_download=False,checkpoint_version='tabicl-regressor-v2-20260212.ckpt',device='cpu',use_amp=False,use_fa3=False,batch_size=1,n_jobs=1,random_state=20261103,offload_mode=False)
    if kind=='TabPFNv2':return TabPFNRegressor.create_default_for_version(ModelVersion.V2,n_estimators=8,auto_scale_n_estimators=False,model_path=BIG/'weights/TabPFN-v2-reg/tabpfn-v2-regressor-v2_default.ckpt',device='cpu',inference_precision=torch.float32,n_preprocessing_jobs=1,random_state=20261103)
    raise ValueError(kind)
def matrices(name,pp):
    features=list(pp[0]['features']);X=np.array([[p['features'][f] for f in features] for p in pp],float)
    result={'speed':(X[:,[0]],features[:1]),'features':(X,features)}
    if name=='mobilised':result['speed_profile']=(X[:,:5],features[:5])
    else:
        result['speed']=(X[:,[features.index('speed')]],['speed'])
    if name in ('older37','older24'):result['features_GDI']=(np.column_stack((X,[p['gdi'] for p in pp])),features+['predicted_GDI'])
    return result
