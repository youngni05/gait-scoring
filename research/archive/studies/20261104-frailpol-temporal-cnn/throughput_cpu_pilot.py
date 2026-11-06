"""Synthetic-only local CPU throughput for the frozen masked architecture; no clinical fit."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,torch
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');sys.path.insert(0,str(H));from native_motion import Model
torch.set_num_threads(2);torch.manual_seed(20261105);rng=np.random.default_rng(20261105);x=rng.normal(size=(64,25,208)).astype(np.float32);x[:,-1]=1;f=rng.normal(size=(64,2)).astype(np.float32);y=rng.normal(size=64).astype(np.float32);net=Model();optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,foreach=False);X=torch.from_numpy(x);F=torch.from_numpy(f);Y=torch.from_numpy(y);times=[]
for block in range(4):
    start=time.perf_counter()
    for step in range(10):
        net.train();optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.smooth_l1_loss(net(X,F),Y,beta=.5);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,foreach=False);optimizer.step()
    times.append(time.perf_counter()-start)
out=B/'throughput-pilot';out.mkdir(exist_ok=True);pt=out/'synthetic-cpu.pt';torch.save(net.state_dict(),pt);np.savez_compressed(out/'synthetic-cpu-inputs.npz',x=x,f=f);net.eval()
with torch.no_grad():original=net(X,F).numpy()
replay=Model().double();replay.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True));replay.eval()
with torch.no_grad():p=replay(torch.from_numpy(x.astype(float)),torch.from_numpy(f.astype(float))).numpy()
assert np.allclose(p,original,atol=1e-6,rtol=1e-6);record=dict(status='passed_synthetic_throughput_only',data='SyntheticGaussian64x25x208+2manualfeatures;nohumanIDs/signals/TUGlabelsused',backend='CPU2threads',model_library_sha256=hashlib.sha256((H/'native_motion.py').read_bytes()).hexdigest(),ten_step_seconds=times,median_warm_seconds_per_step=float(np.median(times[1:])/10),estimated_500_training_steps_seconds=float(np.median(times[1:])*50),cpu64_max_delta=float(np.max(np.abs(p-original))),limits='Localmicrobenchmark,notclinicalaccuracyorstandaloneGPUcomparison. ExistingnativeGPUtrainingcontinuesunchanged;concurrentCPUtasksaffecttiming. Projectionexcludescheckpoint/inferenceoverhead.');(H/'native-motion/synthetic-cpu-throughput.json').write_text(json.dumps(record,indent=2));print(record)
