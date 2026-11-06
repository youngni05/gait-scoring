"""Gate sensitivity training on all10 pilot replays and original B control."""
import datetime,hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BIG=Path('D:/gait-research-data/D_P/normal-label-sensitivity-20261105');OLD=Path('D:/gait-research-data/D_P/gdi-masked-pretrain-v3-20261105')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(HERE/'bundle/protocol.json');snap=read(HERE/'watch/latest-pilot.json');sub=read(HERE/'submission-pilot.json')
for job in sub['jobs'].values():
 row=next(s.split('|') for s in snap['sacct'].splitlines() if s.startswith(job+'|'));assert row[1:3]==['COMPLETED','0:0']
audits=[]
for path in (HERE/'audits').glob('pilot-*.json'):
 a=read(path);assert a['passed'];audits.extend(a['audits'])
assert {(v['family'],v['condition'],v['fold']) for v in audits}=={(f,c,0) for f in p['families'] for c in p['conditions']}
workers={};controls=[]
with np.load(OLD/'data.npz') as z:fold=z['fold'];owners=z['window_record']
pilot=2*int(((fold>=0)&(fold!=0)&(fold!=1))[owners].sum())
for family in p['families']:
 current=BIG/'received/runs/pilot'/family/'B_y100_w1_adult/fold0';previous=OLD/'received/runs/pilot'/family/'B_adult/fold0'
 with np.load(current/'predictions.npz') as a,np.load(previous/'predictions.npz') as b:
  error=max(float(np.max(np.abs(a[k]-b[k]))) for k in ['train_raw','evaluation_raw','prediction']);assert error<.001,error
 # Exact external-sample order/count and original epoch schedule; numeric control tolerance fixed at existing replay threshold.
 c1=read(current/'curve.json');c0=read(previous/'curve.json')
 for a,b in zip(c1,c0):
  for k in ['epoch','optimizer_steps','examples','external_examples','lr']:assert a[k]==b[k]
 controls.append(dict(family=family,original_B_pilot_max_error=error))
 steady=max(read(BIG/'received/runs/pilot'/family/c/'fold0/curve.json')[1]['seconds'] for c in p['conditions'])
 total=sum(4*(steady*(2*int(((fold>=0)&(fold!=f))[owners].sum())/pilot)*p['gdi_epochs'][family][str(f)]+30) for f in range(5))
 minutes=math.ceil(1.5*total/60);workers[family]=dict(estimated_hours=total/3600,time_limit='%02d:%02d:00'%(minutes//60,minutes%60),fits=20,steady_epoch_seconds=steady)
out=dict(authorization='User approved starting proposed label and loss-weight sensitivity; code-only transfer, same previously approved data on same server.',manifest_sha256=sha(HERE/'bundle/manifest.json'),data_sha256=sha(OLD/'data.npz'),pilot_audit_count=10,pilot_jobs=sub['jobs'],workers=workers,control_checks=controls,total_main_fits=40,generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),estimate_basis='Slowest pilot second epoch scaled to full training counts;30s/fit overhead;1.5x time limit')
(HERE/'main-approved.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
