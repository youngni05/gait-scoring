"""Record completion only after the approved 30-person suite is verified."""
import hashlib
import json
import time
from pathlib import Path
from run_ready_subjects import ROOT, OUT, PILOT

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

p=read(OUT/'patterns.json');v=read(OUT/'verification.json');r=read(OUT/'results.json')
m=read(ROOT/'control/add20/experiment-manifest.json');initial=read(ROOT/'control/subset-manifest.json')
d=read(ROOT/'control/add20/download-state.json');d0=read(ROOT/'control/download-state.json')
assert d['status']==d0['status']=='complete' and d['verified_files']==40 and d0['verified_files']==24
assert len(m['files'])==64 and len(v['checks'])==len(r['results'])==p['condition_count']==240
assert p['people']==p['paired_people']==m['selected_subjects']
expected={(s,c,t,w) for s in m['selected_subjects'] for c in ['PG1','PG2']
          for t,widths in [('full',[800,400,200]),('heavy',[800])] for w in widths}
assert {(x['subject'],x['camera'],x['variant'],x['input_width']) for x in r['results']}==expected
assert all(x['reference']['archive_md5_verified'] for x in r['results'])
assert hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()==v['results_sha256']
for e in m['files']:assert (ROOT/'raw'/e['name']).stat().st_size==e['bytes']
assert not list((ROOT/'raw').glob('*.part'))
init=set(initial['selected_subjects']);groups={}
for name,ids in [('initial10',init),('added20',set(p['people'])-init),('all30',set(p['people']))]:
    cs=[x for x in p['comparisons'] if x['subject'] in ids]
    means=[x for x in p['participant_means'] if x['subject'] in ids]
    inside=[x for x in p['both_views_in_frame_diagnostic'] if x['subject'] in ids and x['kept_frames']>0]
    groups[name]={'people':len(ids),'views':len(cs),
        'pg2_higher_people':len(ids&set(p['full_3d_knee_pg2_greater_people'])),
        'common_inside_pg2_higher_people':sum(x['cameras']['PG2']['full_3d_knee_deg']>x['cameras']['PG1']['full_3d_knee_deg'] for x in inside),
        'crop_people':len({x['subject'] for x in cs if x['reference_ankle_outside_image_frame_count']>0}),
        'heavy_better_views':sum(x['heavy_3d_knee_deg']<x['full_3d_knee_deg'] for x in cs),
        'hip_in_top3_views':sum(bool(set(x['top3_position_joints'])&{'LHIP','RHIP'}) for x in cs),
        'full_person_mean_3d':sum(x['full_3d_knee_deg'] for x in means)/len(means),
        'heavy_person_mean_3d':sum(x['heavy_3d_knee_deg'] for x in means)/len(means)}
storage=sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file())
assert storage<21_000_000_000
summary={'status':'complete','completed_unix':time.time(),'raw_files':64,'new_video_files':40,
    'raw_bytes':m['raw_total_bytes'],'new_video_bytes':d['verified_bytes'],'total_storage_bytes':storage,
    'subjects':p['people'],'conditions':240,'cohort_comparison':groups,
    'reference_archives_verified':True,'saved_outputs_verified':True,
    'visual_review':'40 added views reviewed at maximum 2D discrepancy frame; sparse inspection, not framewise manual truth',
    'verification_sha256':hashlib.sha256((OUT/'verification.json').read_bytes()).hexdigest(),
    'results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),
    'limits':['Exploratory laboratory comparison; not clinical joint angles or gait ability scores',
              'No framewise turn/occlusion truth','Joint definitions and synchronization/calibration uncertainty remain']}
write(OUT/'completion.json',summary);write(OUT/'cohort-comparison-final.json',groups)
progress=ROOT/'control/add20/experiment-progress.json';state=read(progress)
state.update(status='complete',updated_unix=time.time(),completed_conditions=240,new_completed_conditions=160,
             completed_people=p['people'],last_verified_snapshot=str(OUT))
write(progress,state)
report=PILOT/'experiment-results.html';html=report.read_text(encoding='utf-8')
marker='<!-- expansion-20261029-start -->';assert html.count(marker)==1
banner=f'<p class="note"><b>30명·240조건 비교 완료.</b> 추가 영상40개(10.057GB)를 포함한 전체64개 원본과 저장 결과를 확인했다. 원본은18.87GB, 좌표 추출본·실험 결과 등을 포함한 D드라이브 저장량은 약{storage/1e9:.2f}GB다. 수치는 연구실 자료의 기하학적 굽힘각 비교이며 임상적 정확성 검증이나 보행 능력 점수가 아니다.</p>'
if '<b>30명·240조건 비교 완료.</b>' not in html:
    report.write_text(html.replace(marker,marker+'\n'+banner,1),encoding='utf-8')
print(json.dumps({'groups':groups,'storage_GB':storage/1e9},indent=2))
