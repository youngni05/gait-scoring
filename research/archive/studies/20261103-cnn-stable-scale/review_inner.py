"""Replay the 40 saved inner checkpoints without fitting or modifying original outputs."""
from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,'D:/Projects/D_P/outputs/motionbert-smoke-20261015/deps')
import torch,numpy as np
H=Path('D:/Projects/D_P/studies/20261103-cnn-stable-scale');B=Path('D:/gait-research-data/D_P/cnn-stable-scale-20261103');ROOT=B/'received/final/original'
sys.path.insert(0,str(ROOT));from model import GdiCNN,channels
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mae(y,p,ids):
    perrow=np.abs(p-y).mean(axis=1)
    return float(np.mean([perrow[ids==i].mean() for i in np.unique(ids)]))
checks=[];start=time.monotonic()
for variant in ['baseline','stable_scale']:
    with np.load(B/'data'/f'{variant}.npz') as z:d={k:z[k] for k in z.files}
    for outer in range(5):
        for inner in range(5):
            if inner==outer:continue
            folder=ROOT/'runs/main'/variant/f'fold{outer}'/f'inner{inner}';ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True);curve=read(folder/'curve.json');assert ck['history']==curve
            tr=(d['fold']!=outer)&(d['fold']!=inner);va=d['fold']==inner;y=d['y'];pid=d['person_id'];assert ck['identity']['train_people']==np.unique(pid[tr]).tolist() and ck['identity']['validation_people']==np.unique(pid[va]).tolist()
            lo=float(ck['lo']);span=float(ck['span']);assert lo==float(np.float32(y[tr].min()));assert span==float(np.float32(max(float(y[tr].max()-y[tr].min()),1e-6)))
            m=GdiCNN();m.load_state_dict(ck['model']);m.eval();pred=[]
            with torch.inference_mode():
                for i in range(0,len(d['x']),64):
                    x=channels(torch.from_numpy(np.array(d['x'][i:i+64],copy=True))).reshape(-1,12,124);pred.append((m(x)[:,0]*span+lo).reshape(-1,2).numpy())
            wp=np.concatenate(pred).astype(float);r=np.zeros((len(y),2));np.add.at(r,d['window_record'],wp);r/=np.bincount(d['window_record'],minlength=len(y))[:,None]
            coef=np.linalg.lstsq(np.column_stack([r[tr].ravel(),np.ones(2*sum(tr))]),y[tr].ravel(),rcond=None)[0];p=r*coef[0]+coef[1];last=curve[-1]
            values={'train_person_mae':mae(y[tr],p[tr],pid[tr]),'train_raw_person_mae':mae(y[tr],r[tr],pid[tr]),'validation_person_mae':mae(y[va],p[va],pid[va]),'validation_raw_person_mae':mae(y[va],r[va],pid[va])}
            differences={k:abs(v-last[k]) for k,v in values.items()};assert max(differences.values())<.001
            correction_delta=float(np.max(abs((r[va]*np.array(last['calibration'])[0]+np.array(last['calibration'])[1])-p[va])));assert correction_delta<.001
            checks.append(dict(variant=variant,outer=outer,inner=inner,checkpoint_sha256=sha(folder/'latest.pt'),metric_differences=differences,training_only_correction_max_difference=correction_delta));print('inner replay',len(checks),variant,outer,inner,max(differences.values()),flush=True)
out=dict(passed=True,models=40,all_training_only_normalization_exact=True,training_only_calibration_rebuilt=True,max_metric_difference=max(max(c['metric_differences'].values()) for c in checks),absolute_tolerance_gdi=.001,seconds=time.monotonic()-start,checks=checks,code_sha256=sha(Path(__file__)))
(H/'inner-verification.json').write_text(json.dumps(out,indent=2),encoding='utf8')
