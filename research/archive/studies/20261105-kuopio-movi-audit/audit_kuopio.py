"""File-level trial linkage and original-marker inspection; no GDI labels generated."""
import sys,json,re,collections,struct,zlib,hashlib,csv,concurrent.futures
from pathlib import Path
from inspect_sources import HERE,BIG,fetch,dump
sys.path.insert(0,str(HERE/'deps'))
import numpy as np
import openpyxl,ezc3d

def read_member(archive,member):
    dest=BIG/'samples'/member['name']
    if dest.exists():
        value=dest.read_bytes()
        if len(value)==member['size'] and zlib.crc32(value)==member['crc32']:return dest
    start=member['offset'];end=min(archive['size']-1,start+member['compressed']+65566)
    raw=fetch(archive['url'],start,end)
    h=struct.unpack('<4s5H3I2H',raw[:30]);assert h[0]==b'PK\x03\x04';offset=30+h[-2]+h[-1]
    data=raw[offset:offset+member['compressed']]
    assert h[3] in (0,8),h
    value=zlib.decompress(data,-15) if h[3]==8 else data
    assert len(value)==member['size'] and zlib.crc32(value)==member['crc32'],member['name']
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(value);return dest

def main():
    archives=[json.loads(p.read_text(encoding='utf-8')) for p in BIG.glob('*.members.json')]
    members={m['name']:(a,m) for a in archives for m in a['members']}
    wb=openpyxl.load_workbook(HERE/'info_participants.xlsx',read_only=True,data_only=True);vals=list(wb.active.values);dem={f'{int(r[0]):02d}':dict(zip(vals[0],r)) for r in vals[1:] if r[0] is not None}
    poses=collections.defaultdict(list);walks={};calibs=collections.defaultdict(list)
    for name,(archive,m) in members.items():
        p=name.split('/');sid=p[0]
        if len(p)<3:continue
        if p[1]=='openpose' and name.endswith('.json'):poses[(sid,p[2].removeprefix('json_'))].append(m)
        if p[1]=='mocap' and name.endswith('.c3d'):
            trial=Path(name).stem
            if re.fullmatch(r'[lr]_(comf|slow|fast)_\d+',trial):walks[(sid,trial)]=m
            else:calibs[sid].append(trial)
    rows=[]
    for key in sorted(set(poses)|set(walks)):
        sid,trial=key;ps=poses.get(key,[]);frames=sorted(int(re.search(r'_(\d+)_keypoints.json$',m['name'])[1]) for m in ps)
        invalid=str(dem[sid].get('Invalid_trials') or '').replace(' ','').split(',')
        rows.append({'person':sid,'trial':trial,'speed':trial.split('_')[1], 'pose_frames':len(ps),'c3d':key in walks,'paired':bool(ps) and key in walks,'provider_invalid':trial in invalid,
                     'frame_indices_contiguous':frames==list(range(len(frames))) if ps else None,'duration_at_100hz_s':len(ps)/100,
                     'frames_after_30hz_resampling':int(np.floor((len(ps)-1)*.3))+1 if ps else 0})
    with (HERE/'kuopio-trials.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    persons=sorted({k[0] for k in poses});eligible=[r for r in rows if r['paired'] and not r['provider_invalid']]
    summary={'all_archive_members':sum(len(a['members']) for a in archives),'mocap_people':len({k[0] for k in walks}),'walking_c3d_files':len(walks),'pose_people':len(persons),'pose_ids':persons,'pose_trials':len(poses),
             'pose_frame_json':sum(len(x) for x in poses.values()),'raw_video_files':[k for k in members if Path(k).suffix.lower() in ['.avi','.mp4','.mov','.mkv']],
             'paired_trials':sum(r['paired'] for r in rows),'paired_provider_valid_trials':len(eligible),'paired_provider_valid_people':len({r['person'] for r in eligible}),
             'paired_valid_by_speed':dict(collections.Counter(r['speed'] for r in eligible)),'longest_pose_frames':max(r['pose_frames'] for r in rows),
             'current_124frame_30hz_eligible':sum(r['frames_after_30hz_resampling']>=124 for r in eligible),
             'noncontiguous_pose_sequences':sum(r['frame_indices_contiguous'] is False for r in rows),'calibrations':dict(calibs),'provider_note_missing_pose_ids':[11,14,37,49],
             'pose_subjects_per_archive':{a['archive']:sorted({m['name'].split('/')[0] for m in a['members'] if '/openpose/' in m['name'] and m['name'].endswith('.json')}) for a in archives}}
    dump(HERE/'kuopio-counts.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['calibrations','pose_subjects_per_archive','pose_ids']}),flush=True)
    # Inspect every participant's skeleton definition and representative raw measurement files.
    selected=[k for k in members if k.endswith('.vsk') or k.endswith('.mp')]
    for sid in ['01','02','18','37','49']:
        selected += [k for k in members if k.startswith(sid+'/mocap/') and k.endswith('.c3d') and ('calib_static.' in k or 'calib_score.' in k)]
        candidates=[r for r in eligible if r['person']==sid and r['speed']=='comf']
        if not candidates:candidates=[r for r in eligible if r['person']==sid]
        if candidates:
            row=candidates[0];selected.append(walks[(sid,row['trial'])]['name'])
            pm=poses[(sid,row['trial'])];selected += [pm[i]['name'] for i in sorted({0,len(pm)//2,len(pm)-1})]
    print('sample files',len(selected),'compressed bytes',sum(members[k][1]['compressed'] for k in selected),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: paths=list(pool.map(lambda k:read_member(*members[k]),selected))
    details=[]
    for path in paths:
        data=path.read_bytes();r={'name':str(path.relative_to(BIG/'samples')),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'zip_crc_verified':True}
        if path.suffix=='.c3d':
            c=ezc3d.c3d(str(path));P=c['parameters'];p=c['data']['points']
            r.update(point_labels=P['POINT']['LABELS']['value'],point_rate=c['header']['points']['frame_rate'],point_frames=p.shape[-1],point_shape=list(p.shape),parameter_groups=list(P),
                     event_parameters={k:(v['value'].tolist() if hasattr(v['value'],'tolist') else v['value']) for k,v in P.get('EVENT',{}).items() if isinstance(v,dict) and 'value' in v})
            r['finite_fraction_per_marker']={n:float(np.isfinite(p[:3,i]).all(axis=0).mean()) for i,n in enumerate(r['point_labels'])}
        elif path.suffix=='.json':
            d=json.loads(data);r.update(people=len(d['people']),pose_lengths=[len(p.get('pose_keypoints_2d',[])) for p in d['people']])
        details.append(r)
    dump(HERE/'kuopio-samples.json',details)
    print('sample inspection complete',len(details),flush=True)

if __name__=='__main__':main()
