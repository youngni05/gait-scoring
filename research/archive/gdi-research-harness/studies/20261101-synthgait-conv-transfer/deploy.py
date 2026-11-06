"""Upload an isolated sealed bundle, submit held and record actual ID before release."""
from pathlib import Path
import subprocess,json,hashlib,zipfile,datetime
ROOT=Path('D:/Projects/D_P');H=ROOT/'gdi-research-harness/studies/20261101-synthgait-conv-transfer'
D=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261101-synthgait-conv-transfer');REMOTE='/abr/RESEARCH_USER/gdi-synth-conv692-20261101'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST';SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST];SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=120)
    if p.returncode:raise RuntimeError(p.stderr+'\n'+p.stdout)
    return p.stdout
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    assert not (H/'submission.json').exists()
    cpu=json.loads((D/'cpu-check-v2/verification.json').read_text());assert cpu['passed']
    manifest=json.loads((H/'manifest.json').read_text());assert cpu['manifest_sha256']==sha(H/'manifest.json')
    for name,value in manifest.items():assert sha(H/name)==value
    (H/'checks').mkdir(exist_ok=True);save(H/'checks/cpu-verification.json',cpu)
    payload=D/'gdi-synth-conv692-20261101.zip';assert not payload.exists()
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
        for name in list(manifest)+['manifest.json','README.md','CNN_LICENSE','SYNTHGAIT_LICENSE','checks/cpu-verification.json']:z.write(H/name,arcname=name)
        z.write(D/'external.npz',arcname='external.npz')
    command(SSH+['test ! -e '+REMOTE+' && test ! -e '+REMOTE+'.zip'])
    data_sha=command(SSH+['sha256sum /abr/RESEARCH_USER/gdi-cnn-person-bmc692-20261031/data.npz']).split()[0]
    assert data_sha==json.loads((H/'protocol.json').read_text())['data_sha256']
    command(SCP+[str(payload),HOST+':'+REMOTE+'.zip'])
    actual=command(SSH+['sha256sum '+REMOTE+'.zip']).split()[0];assert actual==sha(payload)
    command(SSH+['mkdir '+REMOTE+' && cd '+REMOTE+' && /usr/bin/python3.8 -B -m zipfile -e '+REMOTE+'.zip . && mkdir logs && ln -s /abr/RESEARCH_USER/gdi-cnn-person-bmc692-20261031/data.npz data.npz'])
    job=command(SSH+['cd '+REMOTE+' && sbatch --hold --parsable submit.slurm']).strip().split(';')[0];assert job.isdigit()
    submission={'jobs':[job],'remote':REMOTE,'phase':'held_registered','archive':str(payload),'archive_sha256':actual,
        'manifest_sha256':sha(H/'manifest.json'),'gpu_per_job':1,'total_gpus':1,'submitted_at':datetime.datetime.now().astimezone().isoformat()}
    save(H/'submission.json',submission)
    reg=ROOT/'gdi-research-harness/control/gpu-watch/registry.json';r=json.loads(reg.read_text(encoding='utf-8-sig'))
    assert r['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a' and r['automation_id']=='gdi-2'
    assert not any(e['id']=='synthgait-conv-transfer-692' for e in r['experiments'])
    r['experiments'].append({'id':'synthgait-conv-transfer-692','kind':'full_training_pair',
        'local':str(H.relative_to(ROOT)).replace('\\','/'),'remote':REMOTE,'jobs':[job],'preserve_allocations':[],
        'expected_fits':50,'expected_pretraining_fits':25,'expected_outer_folds':10,'variants':['scratch','synth'],
        'run_root':'runs/main','curve_glob':'*/fold*/*/curve.json','outer_complete_glob':'*/fold*/complete.json',
        'summary_path':'runs/pair_summary.json','marker_path':'runs/summary_complete.json','review_state':'pending','reported_digest':None,
        'manifest_sha256':submission['manifest_sha256'],'authorization':'2026-11-01 explicit user: start native short SynthGait convolution-only transfer. Separate new request from previous RankSim follow-up.',
        'audit':'Read own README/protocol/submission.18native16FPS walks14train4val; clip cadence only, no heel-strike loss/resampling/fakeGDI. Transfer only12conv tensors; reset ALL BN/dense/optimizer. Matching before-pretraining seed perGDI fit;25external30epoch fits select on4externalwalks then50GDI/10outer same692/1061/11261/folds/persondraws/MSE/train-onlyOLS. Pilot3externalepoch+2GDI3epoch. GPU/CPU64 <=1e-4 then local SHA/replay gate. waiting_local_pilot_audit: receive pilot-review and use receive_pilot.py; matching approval only after verified. Full receive all50GDI and25external selected/latest/all epochs/logs/control toD, actual SHA receipt, audit.py+stem.audit_external+review_results.py for25external and10GDI CPU64. Preserve exactcoef1e-10 failures; never relax. Source personIDs unavailable; no source subject-disjointness claim. Paired scratch/synth first, historical best6.515817/high68=8.782528 separate. No existing job changes, protected data, LJMU tuning or additional experiment.',
        'storage':{'small_outputs_root':str(H),'large_artifacts_root':str(D)}})
    save(reg,r);print(json.dumps(submission,indent=2))
if __name__=='__main__':main()
