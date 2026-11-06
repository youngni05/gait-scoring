"""The approved original four-condition CUDA pilot; preserve all local protocols."""
from school_gpu_common import *
import datetime,zipfile,re
assert read(H/'external-transfer-approval.json')['status']=='approved_by_user'
assert not (S/'submission.json').exists(),'Inspect existing submission before any retry'
proto=read(H/'protocol.json');original_protocol=sha(H/'protocol.json');assert proto['owner_thread']==OWNER and proto['data_sha256']==sha(B/'data.npz')
assert read(H/'data-verification.json')['protocol_sha256']==original_protocol
S.mkdir(exist_ok=True);proto['training_code_sha256']=sha(H/'bundle/train.py');proto['source_protocol_sha256']=original_protocol;save(S/'protocol.json',proto)
script='''#!/bin/bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd /abr/RESEARCH_USER/frailpol-temporal-cnn-20261104
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/usr/bin/python3.8 -u -B train.py "$@"
'''
(S/'submit.slurm').write_text(script,encoding='utf8',newline='\n');archive=B/'school-gpu/input-bundle.zip';archive.parent.mkdir(exist_ok=True)
members=[(B/'data.npz','data.npz'),(S/'protocol.json','protocol.json'),(H/'bundle/train.py','train.py'),(S/'submit.slurm','submit.slurm')]
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path,name in members:z.write(path,name)
digest=sha(archive)
remote_py("from pathlib import Path\np=Path("+repr(REMOTE)+")\nassert not p.exists(), 'Target exists; inspect before retry'\np.mkdir()\n(p/'logs').mkdir()\n(p/'control').mkdir()\nprint('created')")
scp(archive,'RESEARCH_USER@PRIVATE_HOST:'+REMOTE+'/input-bundle.zip')
source="import hashlib,zipfile\nfrom pathlib import Path\np=Path("+repr(REMOTE)+")\nf=p/'input-bundle.zip'\nassert hashlib.sha256(f.read_bytes()).hexdigest()=="+repr(digest)+"\nwith zipfile.ZipFile(f) as z:\n assert set(z.namelist())=={'data.npz','protocol.json','train.py','submit.slurm'}\n z.extractall(p)\nprint('verified')"
assert remote_py(source)=='verified'
save(S/'transfer.json',dict(status='uploaded_server_sha256_verified',owner_thread=OWNER,remote=REMOTE,archive_sha256=digest,members={name:dict(sha256=sha(path),bytes=path.stat().st_size) for path,name in members},explicit_user_approval=read(H/'external-transfer-approval.json'),source_protocol_unchanged=sha(H/'protocol.json')==original_protocol))
response=ssh('cd '+REMOTE+' && sbatch --parsable --nodelist=DIS02 --time=00:03:00 --job-name=frailpol-motion-pilot submit.slurm pilot')
save(S/'pilot-submit-response.json',dict(response=response));job=response.split(';')[0].strip();assert re.fullmatch(r'\d+',job),response
record=dict(owner_thread=OWNER,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pilot_job=job,remote=REMOTE,input_bundle_sha256=digest,protocol_sha256=sha(S/'protocol.json'),training_code_sha256=sha(H/'bundle/train.py'),requested_seconds=180,estimated_pilot_total_seconds=120,allocation_multiplier=1.5,main_submitted=False,main_gate='Recover all4pilot checkpoints; independently check hashes, GPU identity, frozen persons, training-only normalization, CPU64 replay at original1e-6abs+relative and timing. Only then submit the approved unchanged main conditions. No unrelated job cancellation.',source_local_results='Already completed local DirectML study retained; CUDA experiment is a separate backend run, not independent external clinical validation.')
save(S/'submission.json',record)
reg=read(H/'watch/registry.json');assert reg['thread_id']==OWNER and reg['automation_id']=='frailpol-gpu';assert not any(e['id']=='frailpol-motion-pilot-20261104' for e in reg['experiments'])
reg['experiments'].append(dict(id='frailpol-motion-pilot-20261104',kind='pilot',jobs=[job],remote=REMOTE,local=str(H),received=str(B/'received/pilot'),expected_conditions=4,expected_folds=1,run_root='runs/pilot',summary_path='runs/pilot/summary.json',marker_path='runs/pilot/summary.json',curve_glob='*/curve.json',outer_complete_glob='*/fit_complete.json',review_state='pending',reported_digest=None,protected_jobs=['34721','34722','32844']))
reg['monitor_estimate']=dict(status='approved_real_pilot_submitted',real_data_training_approval='explicitly_approved',basis='120s provisional pilot total;180s allocation. Pilot measured in active turn before main estimates and heartbeat interval are fixed. No half-remaining or fixed30minute fallback.')
save(H/'watch/registry.json',reg);assert sha(H/'protocol.json')==original_protocol;print(json.dumps(record,indent=2))
