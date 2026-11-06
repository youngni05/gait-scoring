import os,sys,json,importlib.util,argparse,time
from pathlib import Path
if os.name=='nt':sys.path[:0]=['D:/Projects/D_P/outputs/motionbert-smoke-20261015/deps']
import numpy as np,torch
import run
from settings import NF,NT,V,VARIANT
from model import GdiCNN,channels
from windows import WindowSet
ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--synthetic-full',action='store_true');a=ap.parse_args()
c=run.config();run.setup(1234,a.device);d=run.data();a.output.mkdir(parents=True,exist_ok=True)
assert d['x'].values.shape==(11261,2,124,12)
run.smoke(c,a.output/'smoke',a.device)
proof={'variant':VARIANT,'real_data_optimizer_steps':0,'people':len(np.unique(d['person_id'])),'records':len(d['y']),'windows':len(d['x'].values),'shape':list(d['x'].values.shape),'all20_person_splits_checked':True,'removed_feature_invariance_checked':NF==8}
if VARIANT in ['baseline','stable_scale']:
 spec=importlib.util.spec_from_file_location('original_model',run.ROOT/'original/paper_model.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 torch.manual_seed(798);m=GdiCNN().to(a.device);torch.manual_seed(798);o=old.GdiCNN().to(a.device)
 assert all(torch.equal(v,o.state_dict()[k]) for k,v in m.state_dict().items())
 m.eval();o.eval();x=channels(torch.from_numpy(d['x'].values[:3])).reshape(-1,NF,NT).to(a.device)
 with torch.no_grad():diff=float((m(x)-o(x)).abs().max())
 assert diff==0;proof['original_full124_forward_max_error']=diff
if a.synthetic_full:
 rng=np.random.default_rng(12);n=10;owner=np.repeat(np.arange(n),2)
 sd=dict(x=WindowSet(rng.normal(size=(20,2,NT,NF)).astype('float32'),owner,n),y=rng.uniform(50,100,(n,2)),person_id=np.array(['fake%d'%i for i in range(n)]),video_id=np.array(['fakev%d'%i for i in range(n)]),fold=np.arange(n)%5,cohort=np.arange(n)%2,original_row_index=np.arange(n))
 sc={**c,'max_epochs':2,'batch_size':32}
 for f in range(5):run.train_fold(sd,sc,f,a.output/'synthetic_full',a.device,None)
 run.summarize(sd,sc,a.output/'synthetic_full')
 proof['synthetic_nested_fits']=25;proof['synthetic_outer_predictions']=5
run.save(a.output/'check.json',proof);print(json.dumps(proof),flush=True)
