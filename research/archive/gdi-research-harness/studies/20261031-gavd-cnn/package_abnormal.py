"""Package visually screened abnormal candidates; keep source labels unmodified."""
import hashlib,json,tarfile
from audit_gavd import HERE,DATA,write_csv,write_json
from prepare_comparison import read_csv
rows=read_csv(HERE/'abnormal_screening.csv');meta=json.loads((DATA/'video_inventory.json').read_text())
reject={3:'source frame range exceeds decoded video; timing mismatch',11:'view label lateral but sampled back/oblique view',25:'scene/view changes within annotation',26:'long sequence with distant target and moving camera',31:'treadmill and body support unmatched to normal overground',34:'moving distant camera and target too small',36:'on-screen Normal Gait contradicts abnormal source label',37:'split-screen before/after and actual 24 FPS unmatched'}
audit=[];kept=[];jobs=[]
for i,r in enumerate(rows,1):
    reason=reject.get(i,'')
    if r['gait_pattern']=='exercise':reason='exercise protocol excluded from overground first comparison'
    m=meta[r['video_id']];first=int(r['annotation_frame_first'])-1;last=min(int(r['annotation_frame_last'])-1,first+1498)
    if last>=m['frames']:reason='source annotation extends beyond decoded video'
    audit.append(dict(index=i,record_id=r['record_id'],video_id=r['video_id'],included_for_pose=not bool(reason),reason=reason or 'sampled visual check passed; pose and continuous review pending',actual_fps=m['fps']))
    if reason:continue
    kept.append(r);jobs.append(dict(record_id=r['record_id'],video_id=r['video_id'],video_file='videos/'+r['video_id']+'.mp4',video_sha256=m['sha256'],decoded_first=first,decoded_last=last,annotation_offset=-1,width=m['width'],height=m['height'],fps=m['fps'],view=r['view'],label=r['label']))
write_csv(HERE/'abnormal_visual_audit.csv',audit);write_csv(HERE/'abnormal_pose_candidates.csv',kept);write_json(HERE/'abnormal_jobs.json',dict(records=jobs,score_blind=True))
archive=DATA/'transfer/gavd-abnormal-pilot.tar'
if archive.exists():raise ValueError('No overwrite')
entries=[(HERE/'abnormal_jobs.json','abnormal_jobs.json')]+[(DATA/'videos'/(j['video_id']+'.mp4'),j['video_file']) for j in jobs]
with tarfile.open(archive,'w') as t:
    for p,n in entries:t.add(p,arcname=n)
report=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),records=len(jobs),pilot_frames=3*len(jobs),full_frames=sum(j['decoded_last']-j['decoded_first']+1 for j in jobs))
write_json(HERE/'abnormal_transfer_manifest.json',report);print(json.dumps(report))
