"""Seal the public 20-video BODY25 inference-only payload; no GDI data."""
from pathlib import Path
import json,hashlib,tarfile,ast
ROOT=Path('D:/Projects/D_P')
HERE=ROOT/'gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20'
DATA=Path('D:/gait-research-data/D_P/synthgait19k-review/sagittal20')
OLD=ROOT/'gdi-research-harness/studies/20261031-gavd-cnn'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=json.loads((HERE/'download-manifest.json').read_text(encoding='utf-8'))
    inv=json.loads((HERE/'video-inventory.json').read_text(encoding='utf-8'))
    assert source['complete'] and inv['all20_decoded']
    lengths={r['sample_id']:r for r in inv['records']};jobs=[]
    for r in source['records']:
        m=lengths[r['sample_id']];f=r['files']['.mp4'];assert sha(Path(f['path']))==f['sha256']
        jobs.append(dict(record_id=r['sample_id'],video_id=r['walk_id'],video_file='videos/'+Path(f['path']).name,
            video_sha256=f['sha256'],decoded_first=0,decoded_last=m['frame_count']-1,annotation_offset=0,
            width=832,height=480,fps=16,view='sagittal',split=r['split']))
    (HERE/'pose_jobs.json').write_text(json.dumps({'records':jobs,'inference_only':True},indent=2),encoding='utf-8')
    for name in ['gpu_pose_worker.py','pose_torch.py']:
        (HERE/name).write_bytes((OLD/name).read_bytes());ast.parse((HERE/name).read_text(encoding='utf-8'),feature_version=(3,8))
    script='''#!/usr/bin/env bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --exclude=DIS04,DIS05,DIS06
#SBATCH --job-name=synth-body25
#SBATCH --output=logs/worker-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec /usr/bin/env /usr/bin/python3.8 -u -B gpu_pose_worker.py --mode full --manifest pose_jobs.json
'''
    (HERE/'pose.slurm').write_text(script,encoding='utf-8',newline='\n')
    entries=[(HERE/n,n) for n in ['gpu_pose_worker.py','pose_torch.py','pose_jobs.json','pose.slurm']]
    entries += [(Path(r['files']['.mp4']['path']),'videos/'+Path(r['files']['.mp4']['path']).name) for r in source['records']]
    probes=list(sorted((DATA/'official_probe').glob('*.npz')));assert len(probes)==3
    entries += [(p,'official_probe/'+p.name) for p in probes]
    archive=DATA/'synth-pose20.tar'
    assert not archive.exists(),'Do not overwrite an existing sealed transfer'
    with tarfile.open(archive,'w') as tar:
        for p,name in entries:tar.add(p,arcname=name)
    payload={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'expected_frames':sum(m['frame_count'] for m in inv['records']),
        'expected_records':20,'remote':'/abr/RESEARCH_USER/synthgait-pose20-20261101','inference_only':True,'optimizer_steps':0,
        'files':[dict(path=n,sha256=sha(p),bytes=p.stat().st_size) for p,n in entries],
        'network':{'network.json':'364c8bd15b82a6eaaf3fc61ac2e966a45a57ba65a2ff1e3d65a42ae56d1118c3',
                   'network_weights.npz':'cd0bee115884ddda742b513c139c282432ff0707cadee8a8a0795e53736fb686'}}
    (HERE/'pose-transfer.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k!='files'},indent=2))
if __name__=='__main__':main()
