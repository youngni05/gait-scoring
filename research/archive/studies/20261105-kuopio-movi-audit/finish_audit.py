"""Rebuild report counts and retain original model/angle-field evidence."""
import sys,json,csv,io,tarfile,collections,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/kuopio-movi-audit-20261105')
sys.path.insert(0,str(HERE/'deps'))
sys.path.insert(0,'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')
import ezc3d,numpy as np
from scipy.io import loadmat
def dump(name,value): (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    provider=json.loads((HERE/'movi-provider-current.json').read_text(encoding='utf-8'))['data']['latestVersion']['files']
    files={f['label']:f['dataFile'] for f in provider}
    a=json.loads((HERE/'movi-audit.json').read_text());rows=a['subjects'];assert len(rows)==len({r['id'] for r in rows})==90
    for archive in a['archives']:assert archive['md5']==files[archive['name']]['checksum']['value']
    counts={};selected=[]
    for cam in ['PG1','PG2']:
        views=[(r,next(v for v in r['views'] if v['camera']==cam)) for r in rows]
        assert all(v['listed']==(v['video'] in files) for r,v in views)
        counts[cam]={'available_videos':sum(v['listed'] for r,v in views),'missing_video_ids':[r['id'] for r,v in views if not v['listed']]}
        for n in [30,60,124]:
            passed=[(r,v) for r,v in views if v['listed'] and not r['missing_marker_names'] and v['max_contiguous_frames']>=n]
            intervals=[(r,v,s) for r,v in passed for s in v['intervals'] if s['frames']>=n]
            counts[cam][str(n)]={'people':len(passed),'videos':len(passed),'intervals':len(intervals),'ids':sorted(r['id'] for r,v in passed),'age_min':min([r['age'] for r,v in passed],default=None),'age_max':max([r['age'] for r,v in passed],default=None)}
            if n==30:
                for r,v,s in intervals:selected.append({'person':r['id'],'age':r['age'],'camera':cam,'video':v['video'],**s,'nominal_duration_s':s['frames']/30,'passes_60_frames':s['frames']>=60,'full_gait_cycle_verified':False,'gdi_label_verified':False})
    with (HERE/'movi-selected-intervals.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    counts['all_round_F_video_inventory']={cam:sum(k.startswith('F_'+cam+'_Subject_') and k.endswith('.avi') for k in files) for cam in ['PG1','PG2','CP1','CP2']}
    counts['all_round_F_video_inventory_all_extensions']={cam:sum(k.startswith('F_'+cam+'_Subject_') for k in files) for cam in ['PG1','PG2','CP1','CP2']}
    counts['excluded_marker_names']={r['id']:r['missing_marker_names']+r['missing_view_markers'] for r in rows if r['missing_marker_names'] or r['missing_view_markers']}
    counts['age_range_all']=[min(r['age'] for r in rows),max(r['age'] for r in rows)]
    dump('movi-counts.json',counts)

    models=[]
    for p in sorted((BIG/'samples').rglob('*.vsk')):
        root=ET.parse(p).getroot();markers=[x.attrib for x in root.iter('Marker')];segments=[x.attrib for x in root.iter('Segment')]
        models.append({'person':int(p.stem),'model':root.get('MODEL'),'marker_names':[x.get('NAME') for x in markers],'segment_names':[x.get('NAME') for x in segments],'pelvis_markers':[x for x in markers if 'Pelvis' in x.get('NAME','')]})
    c3ds=[]
    for p in sorted((BIG/'samples').rglob('*.c3d')):
        c=ezc3d.c3d(str(p));params=c['parameters']['POINT'];labels=list(params['LABELS']['value']);desc=list(params['DESCRIPTIONS']['value'])
        c3ds.append({'file':str(p.relative_to(BIG)),'frames':c['data']['points'].shape[-1],'rate':float(params['RATE']['value'][0]),'user_model_labels':[{ 'label':n,'description':desc[i]} for i,n in enumerate(labels) if 'USERMO' in n],
                     'angle_unit_metadata':{k:list(v['value']) for k,v in params.items() if isinstance(v,dict) and ('USERMO' in k or k in ['ANGLE_UNITS','UNITS'])},'event_count':int(c['parameters'].get('EVENT',{}).get('USED',{'value':[0]})['value'][0])})
    with tarfile.open('D:/gait-research-data/MoVi/raw/F_Subjects_1_45.tar') as t:
        member=next(x for x in t if Path(x.name).name=='F_v3d_Subject_1.mat');m=loadmat(io.BytesIO(t.extractfile(member).read()),simplify_cells=True)['Subject_1_F']['move']
        markers=[{'name':str(n),'description':str(d)} for n,d in zip(m['markerName'],m['markerDescription'])]
        fields={k:list(m[k].shape) for k in ['markerLocation','virtualMarkerLocation','jointsAffine_v3d','jointsExpMaps_v3d','jointsTranslation_v3d']}
    dump('angle-source-evidence.json',{'kuopio_models':models,'kuopio_c3d_samples':c3ds,'movi_example_shapes':fields,'movi_example_marker_descriptions':markers,'clinical_angle_convention_equivalence_verified':False})

    trials=list(csv.DictReader((HERE/'kuopio-trials.csv').open(encoding='utf-8-sig')));kc=json.loads((HERE/'kuopio-counts.json').read_text())
    assert len(trials)==kc['walking_c3d_files']==3193
    assert sum(r['paired']=='True' for r in trials)==kc['pose_trials']==3003
    valid=[r for r in trials if r['paired']=='True' and r['provider_invalid']=='False']
    assert len(valid)==kc['paired_provider_valid_trials']==2885
    assert len({r['person'] for r in valid})==49
    samples=json.loads((HERE/'kuopio-samples.json').read_text());assert len(models)==51
    for sample in samples:
        p=BIG/'samples'/Path(sample['name'].replace('\\','/'))
        assert sample['zip_crc_verified'] and hashlib.sha256(p.read_bytes()).hexdigest()==sample['sha256']
    videos=json.loads((HERE/'movi-video-verification.json').read_text())
    assert len(videos)==counts['PG1']['available_videos']==87
    assert {v['id'] for v in videos}=={r['id'] for r in rows if next(v for v in r['views'] if v['camera']=='PG1')['listed']}
    for v in videos:
        assert v['decoded_frames']==v['walking_end_one_based']-v['walking_start_one_based']+1
        assert hashlib.sha256((BIG/'movi-video-ranges'/f"{v['id']:02d}-walking.bin").read_bytes()).hexdigest()==v['range_sha256']
    dump('verification.json',{'passed':True,'kuopio_model_files':len(models),'kuopio_crc_and_sha256_verified_samples':len(samples),'kuopio_paired_trials':sum(r['paired']=='True' for r in trials),'kuopio_conservative_trials':len(valid),'movi_original_mats':len(rows),'movi_original_archive_checksums_match_current_provider':True,'movi_pg1_videos_decoded':len(videos),'movi_walking_frames_decoded':sum(v['decoded_frames'] for v in videos),'gdi_labels_computed':0,'complete_gait_cycles_verified':0,'video_source_range_hashes_verified':True})
    print(json.dumps(counts,ensure_ascii=False))
if __name__=='__main__':main()
