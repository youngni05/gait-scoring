"""Gate main submission on completed/replayed pilots and measured timing."""
import datetime,hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/gdi-masked-pretrain-v3-20261105')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(HERE/'bundle/protocol.json');snap=read(HERE/'watch/latest-pilot.json');sub=read(HERE/'submission-pilot.json')
for job in sub['jobs'].values():
    row=next(s.split('|') for s in snap['sacct'].splitlines() if s.startswith(job+'|'));assert row[1:3]==['COMPLETED','0:0']
assert len(snap['fits'])==12 and all(r['complete'] for r in snap['fits'])
audits=[]
for path in (HERE/'audits').glob('pilot-*.json'):
    a=read(path);assert a['passed'];audits.extend(a['audits'])
assert {(v['family'],v['condition'],v['fold']) for v in audits}=={(f,c,0) for f in p['families'] for c in p['conditions']}
assert all(v['passed'] and v['server_seed_reconstruction_passed'] for v in audits)
with np.load(BIG/'data.npz') as z:fold=z['fold'];owners=z['window_record']
pilot=(fold>=0)&(fold!=0)&(fold!=1);pilot_examples=int(pilot[owners].sum())*2
workers={};expected=[]
for family in p['families']:
    seconds=[];pretrain=[]
    for condition in p['conditions']:
        folder=BIG/'received/runs/pilot'/family/condition/'fold0';curve=read(folder/'curve.json');seconds.append(curve[1]['seconds'])
        if (folder/'pretrain-curve.json').exists():pretrain.append(read(folder/'pretrain-curve.json')[-1]['seconds']/40)
    # Slowest observed steady second epoch, scaled by exact Stanford example count.
    steady=max(seconds);pre_step=max(pretrain);total=0
    for f in range(5):
        examples=2*int(((fold>=0)&(fold!=f))[owners].sum());epochs=p['gdi_epochs'][family][str(f)]
        for condition in p['conditions']:
            ssl=condition=='C' or condition.startswith('D_')
            estimate=steady*(examples/pilot_examples)*epochs+(p['pretrain_steps']*pre_step if ssl else 0)+30
            expected.append(dict(family=family,condition=condition,fold=f,epochs=epochs,estimated_seconds=estimate));total+=estimate
    # 30s/fit covers initialization, sampling plan, final prediction/checkpoint and launch.
    minutes=math.ceil(total*1.5/60);workers[family]=dict(estimated_hours=total/3600,time_limit='%02d:%02d:00'%(minutes//60,minutes%60),fits=30,pretraining_fits=15,steady_epoch_seconds=steady,pretrain_seconds_per_update=pre_step)
out=dict(authorization='User explicitly approved this40MB data/mask/code payload school GPU transfer, A-D pilots and main training. Revisions change only deterministic interpolation and initialization control; data SHA unchanged.',
    manifest_sha256=sha(HERE/'bundle/manifest.json'),data_sha256=sha(BIG/'data.npz'),pilot_jobs=sub['jobs'],pilot_audit_count=len(audits),
    calibrated_cpu_max_error=max(v['calibrated_cpu_replay_max_error'] for v in audits),local_cross_version_initialization_mismatch_preserved=any(not v['local_cross_version_seed_sha_matches'] for v in audits),
    workers=workers,fits=expected,total_main_fits=60,total_pretraining_fits=30,
    estimate_basis='Maximum observed second-epoch time scaled to each outer training example count; maximum observed pretraining time/update;30s/fit fixed overhead; Slurm limit1.5x total. No hold after completion.',
    generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(HERE/'main-approved.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['fits','authorization']},indent=2))
