from pathlib import Path
import os,sys,json,hashlib
HERE=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');BIG=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103');OLD=Path('D:/gait-research-data/D_P/sppb-tabular-foundation-20261103')
for k,v in {'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HOME':str(BIG/'cache/hf'),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1','TABPFN_ALLOW_CPU_LARGE_DATASET':'1','SKB_DATA_DIRECTORY':str(BIG/'cache/skrub'),'MPLCONFIGDIR':str(BIG/'cache/mpl'),'TABPFN_MODEL_CACHE_DIR':str(BIG/'weights')}.items():os.environ[k]=v
sys.stdout.reconfigure(encoding='utf8')
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
from tabicl import TabICLClassifier,TabICLRegressor
from tabpfn import TabPFNClassifier,TabPFNRegressor
from tabpfn.constants import ModelVersion
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def factory(kind,classification):
    if kind=='TabICLv2':
        name='tabicl-'+('classifier' if classification else 'regressor')+'-v2-20260212.ckpt';path=BIG/'weights'/name if classification else OLD/'weights/TabICL'/name
        return (TabICLClassifier if classification else TabICLRegressor)(n_estimators=8,model_path=path,allow_auto_download=False,checkpoint_version=name,device='cpu',use_amp=False,use_fa3=False,batch_size=1,n_jobs=1,random_state=20261103,offload_mode=False)
    name='tabpfn-v2-'+('classifier' if classification else 'regressor')+'-v2_default.ckpt';path=BIG/'weights'/name if classification else OLD/'weights/TabPFN-v2-reg'/name
    return (TabPFNClassifier if classification else TabPFNRegressor).create_default_for_version(ModelVersion.V2,n_estimators=8,auto_scale_n_estimators=False,model_path=path,device='cpu',inference_precision=torch.float32,n_preprocessing_jobs=1,random_state=20261103)
def transform(y,target):return np.log(y) if target=='tug' else y
def undo(y,target):return np.exp(y) if target=='tug' else y
