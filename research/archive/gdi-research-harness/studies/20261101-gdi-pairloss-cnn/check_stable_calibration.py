"""Compare proposed same-OLS arithmetic on identical saved server/local pilot bytes."""
from pathlib import Path
import json,hashlib
import numpy as np
import manage
from stable_calibration import stable_calibrate
H=Path(__file__).resolve().parent

def main():
    code=(H/'stable_calibration.py').read_text(encoding='utf-8')
    remote_source=code+'''\nfrom pathlib import Path
import json,hashlib
p=Path(ROOT)/'seed1/runs/pilot/fit/raw_predictions'
with np.load(Path(ROOT)/'data.npz',allow_pickle=False) as z:
 y=z['y'][(z['fold']!=0)&(z['fold']!=1)]
out=[]
for epoch in (1,2,3):
 q=p/('epoch%04d.npz'%epoch)
 with np.load(q) as z:coef=stable_calibrate(z['train_raw'],y)
 out.append({'epoch':epoch,'coef':coef.tolist(),'raw_sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
print(json.dumps(out))
'''
    remote=json.loads(manage.remote(remote_source))
    with np.load(manage.INPUT,allow_pickle=False) as z:y=z['y'][(z['fold']!=0)&(z['fold']!=1)]
    comparisons=[]
    root=manage.DATA/'received/pilot/seed1/unpacked/runs/pilot/fit/raw_predictions'
    for r in remote:
        path=root/('epoch%04d.npz'%r['epoch']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['raw_sha256']
        with np.load(path) as z:
            coef=stable_calibrate(z['train_raw'],y)
            # Orthogonality checks verify OLS, independent of coefficient matching.
            residual=z['train_raw'].ravel()*coef[0]+coef[1]-y.ravel()
            normal=float(abs(residual.sum()));centered=float(abs(((z['train_raw'].ravel()-z['train_raw'].mean())*residual).sum()))
        error=float(abs(coef-np.asarray(r['coef'])).max());assert error<1e-10
        comparisons.append(dict(epoch=r['epoch'],maximum_server_local_coefficient_difference=error,normal_equation_residual_sum=normal,centered_normal_equation_residual=centered))
    result=dict(passed=True,training_or_inference_rerun=False,same_saved_pilot_bytes=True,
                calibration_definition='Unchanged pooled training-only OLS; centered fsum evaluation',
                maximum_difference=max(r['maximum_server_local_coefficient_difference'] for r in comparisons),
                original_tolerance=1e-10,comparisons=comparisons)
    (H/'checks/stable-calibration-cross-platform.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
