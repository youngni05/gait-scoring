"""Isolated, inference-only school GPU job. Existing jobs are read, never altered."""
import sys,json,tarfile,hashlib,subprocess,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BIG=Path('D:/gait-research-data/D_P/self-video-gdi-20261101');REMOTE='/abr/RESEARCH_USER/self-video-gdi-20261101'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST';SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST];SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args,timeout=120):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
    return p.stdout
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')
def main():
    assert not (HERE/'submission.json').exists(),'Inspect existing submission first'
    BIG.mkdir(parents=True,exist_ok=True);old=HERE.parent/'20261031-gavd-cnn'
    rows=json.loads((HERE/'inventory.json').read_text(encoding='utf8'));jobs=[]
    for r in rows:
        jobs.append(dict(record_id=r['id'],video_id=r['id'],video_file='videos/'+r['id']+'.mp4',video_sha256=r['sha256'],decoded_first=0,decoded_last=r['frames']-1,annotation_offset=0,width=r['width'],height=r['height']))
    save(HERE/'pose_jobs.json',dict(records=jobs,inference_only=True,optimizer_steps=0))
    script='''#!/usr/bin/env bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --exclude=DIS04,DIS05,DIS06
#SBATCH --job-name=self-video-body25
#SBATCH --output=logs/worker-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec /usr/bin/env /usr/bin/python3.8 -u -B gpu_pose_worker.py --mode full --manifest pose_jobs.json
'''
    (HERE/'pose.slurm').write_text(script,encoding='utf8',newline='\n')
    entries=[(old/n,n) for n in ['gpu_pose_worker.py','pose_torch.py']]+[(HERE/n,n) for n in ['pose_jobs.json','pose.slurm']]
    entries += [(Path(r['file']),'videos/'+r['id']+'.mp4') for r in rows]
    probes=Path('D:/gait-research-data/D_P/datasets/GAVD/official_probe')
    entries += [(p,'official_probe/'+p.name) for p in sorted(probes.glob('*.npz'))]
    assert sum(name.startswith('official_probe/') for _,name in entries)==3
    archive=BIG/'inference-payload.tar';assert not archive.exists()
    with tarfile.open(archive,'w') as t:
        for p,n in entries:t.add(p,arcname=n)
    save(HERE/'transfer.json',dict(remote=REMOTE,archive=str(archive),sha256=sha(archive),files=[dict(name=n,sha256=sha(p)) for p,n in entries]))
    before=command(SSH+['squeue -u RESEARCH_USER -o "%.18i %.24j %.2t %.10M %R"'])
    command(SSH+['test ! -e '+REMOTE+' && test ! -e '+REMOTE+'.tar'])
    command(SCP+[str(archive),HOST+':'+REMOTE+'.tar'])
    assert command(SSH+['sha256sum '+REMOTE+'.tar']).split()[0]==sha(archive)
    command(SSH+['mkdir '+REMOTE+' && tar -xf '+REMOTE+'.tar -C '+REMOTE+' && mkdir '+REMOTE+'/logs && ln -s /abr/RESEARCH_USER/gavd-cnn-20261031/network '+REMOTE+'/network'])
    response=command(SSH+['cd '+REMOTE+' && sbatch --hold --parsable pose.slurm']).strip();job=response.split(';')[0];assert job.isdigit()
    save(HERE/'submission.json',dict(job_id=job,remote=REMOTE,held=True,expected_frames=sum(r['frames'] for r in rows),inference_only=True,optimizer_steps=0,before=before,at=datetime.datetime.now().astimezone().isoformat()))
    command(SSH+['scontrol release '+job]);print('SUBMITTED',job,flush=True)
if __name__=='__main__':main()
