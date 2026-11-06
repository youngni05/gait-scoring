"""Inventory the local code and visit-metadata bundle; no network or remote actions."""
from pathlib import Path
import hashlib,json,tarfile
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    paths=[]
    for arm in (0,1):
        b=H/('seed%d'%arm);manifest=json.loads((b/'manifest.json').read_text(encoding='utf-8'))
        for name,value in manifest.items():assert sha(b/name)==value
        paths += [(p,'seed%d/'%arm+p.name) for p in sorted(b.iterdir()) if p.is_file()]
    paths.append((H/'protocol.json','protocol.json'))
    archive=H/'bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p,name in paths:tf.add(p,arcname=name)
    metadata=json.loads((H/'seed0/visit_metadata.json').read_text(encoding='utf-8'))
    report=dict(archive=str(archive),bytes=archive.stat().st_size,archive_sha256=sha(archive),
                contents=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for p,name in paths],
                visit_metadata_fields=['person_id','video_id','age'],metadata_rows_per_arm=len(metadata),
                metadata_source='Existing public de-identified Stanford mobile-gaitlab development records; no new private records',
                contains_raw_GDI_labels=False,contains_coordinate_arrays=False,contains_original_video=False,
                contains_credentials=False,
                planned_destination='ABRM02:/abr/RESEARCH_USER/gdi-pairloss-cnn-20261101',
                existing_remote_input='/abr/RESEARCH_USER/gdi-cnn-multiwindow-20261030/data.npz',
                existing_input_sha256=json.loads((H/'protocol.json').read_text(encoding='utf-8'))['data_sha256'],
                external_transfer_performed=False,slurm_jobs_submitted=False,
                approval_review='Deployment rejected: explicit permission for this metadata payload and destination required')
    (H/'transfer-review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='contents'},indent=2))
if __name__=='__main__':main()
