"""Pin normal pilot jobs and package only public GAVD inputs and probe code."""
import ast,csv,hashlib,json,tarfile
from pathlib import Path
from audit_gavd import HERE,DATA,write_json
from prepare_comparison import ROOT,read_csv

def main():
    rows=[r for r in read_csv(HERE/'review_template.csv') if r['label']=='Normal Gait']
    metadata=json.loads((DATA/'video_inventory.json').read_text(encoding='utf-8'))
    jobs=[]
    for r in rows:
        m=metadata[r['video_id']];first=int(r['annotation_frame_first'])-1
        last=min(int(r['annotation_frame_last'])-1,first+1498)
        assert first>=0 and last<m['frames'] and last-first+1>=124
        jobs.append(dict(record_id=r['record_id'],video_id=r['video_id'],video_file='videos/'+r['video_id']+'.mp4',
                         video_sha256=m['sha256'],decoded_first=first,decoded_last=last,annotation_offset=-1,
                         width=m['width'],height=m['height'],fps=m['fps'],view=r['view'],label=r['label']))
    write_json(HERE/'normal_jobs.json',dict(records=jobs,frame_convention='annotation frame minus 1; visually checked sampled bbox correspondence; exact 0/1 convention not established from source docs',
               actual_GDI_labels=False,normal_label_not_clinical_diagnosis=True))
    existing=ROOT/'gdi-research-harness/studies/20261031-cnn-high-gdi/research/pose_torch.py'
    (HERE/'pose_torch.py').write_bytes(existing.read_bytes())
    archive=DATA/'transfer'/'gavd-normal-pilot.tar';archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():raise ValueError('Archive already exists')
    entries=[(HERE/n,n) for n in ['gpu_pose_worker.py','pose_torch.py','pose.slurm','normal_jobs.json']]
    entries += [(DATA/'videos'/(v+'.mp4'),'videos/'+v+'.mp4') for v in sorted({r['video_id'] for r in jobs})]
    probes=sorted((DATA/'official_probe').glob('*.npz'));assert len(probes)==3,'CPU probes must be complete'
    entries += [(p,'official_probe/'+p.name) for p in probes]
    for p,name in entries:
        if name.endswith('.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    with tarfile.open(archive,'w') as tar:
        for p,name in entries:tar.add(p,arcname=name)
    manifest=dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  bytes=archive.stat().st_size,files=[dict(path=n,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,n in entries],
                  records=len(jobs),pilot_frames=3*len(jobs),full_frames=sum(r['decoded_last']-r['decoded_first']+1 for r in jobs),
                  remote='/abr/RESEARCH_USER/gavd-cnn-20261031',source='Public GAVD candidate videos only; no private development data')
    write_json(HERE/'transfer_manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))

if __name__=='__main__':main()
