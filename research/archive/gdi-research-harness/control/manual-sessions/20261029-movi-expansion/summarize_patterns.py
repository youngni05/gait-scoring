"""Descriptive participant/view comparisons of the frozen baseline; no new thresholds."""
import argparse
import json
from collections import Counter
from datetime import datetime,timezone,timedelta
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse,unquote
from run_ready_subjects import OUT,PILOT,ROOT,digest,np,NAMES

parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=OUT/'results.json')
parser.add_argument('--verification',type=Path,default=OUT/'verification.json')
parser.add_argument('--patterns',type=Path,default=OUT/'patterns.json')
args=parser.parse_args()
rows=json.loads(args.results.read_text(encoding='utf-8'))['results']
verification=json.loads(args.verification.read_text(encoding='utf-8'))
assert len(verification['checks'])==len(rows)
if verification.get('results_sha256'):assert digest(args.results)==verification['results_sha256']
index={(r['subject'],r['camera'],r['variant'],r['input_width']):r for r in rows}
for r in rows:
    prediction=OUT/f"subject-{r['subject']:02d}"/r['variant']/f"{r['camera']}_{r['input_width']}.npz"
    assert digest(prediction)==r['prediction_sha256']
base=sorted([r for r in rows if r['variant']=='full' and r['input_width']==800],key=lambda r:(r['subject'],r['camera']))
people=sorted({r['subject'] for r in base})
paired=[s for s in people if all((s,c,'full',800) in index for c in ['PG1','PG2'])]
def knee(row,kind='pred3d_vs_ref3d'):return row['knee_abs_angle_difference_deg'][kind]['mean']
def mean(values):return sum(values)/len(values) if values else None
comparisons=[]
for r in base:
    s,c=r['subject'],r['camera'];heavy=index.get((s,c,'heavy',800));low=index.get((s,c,'full',200))
    with np.load(OUT/f'subject-{s:02d}/full/{c}_800_comparison.npz') as data:
        ankles=data['reference_xy'][:,[NAMES.index('LAJC'),NAMES.index('RAJC')]]
        finite=np.isfinite(ankles).all(axis=-1)
        outside=finite&((ankles[...,0]<0)|(ankles[...,0]>=800)|(ankles[...,1]<0)|(ankles[...,1]>=600))
        outside_frames=data['frames'][outside.any(axis=1)].tolist()
    comparisons.append({'subject':s,'camera':c,'frames':r['evaluation_frames'],
        'full_position_px':r['position_discrepancy_px']['mean'],'full_3d_knee_deg':knee(r),
        'heavy_3d_knee_deg':knee(heavy) if heavy else None,
        'reference_2d_3d_projection_gap_deg':knee(r,'ref2d_vs_ref3d_projection_difference'),
        'reference_ankle_outside_image_frames':outside_frames,
        'reference_ankle_outside_image_frame_count':len(outside_frames),
        'full_3d_knee_visibility_ge_05_deg':r['knee_visibility_ge_05']['pred3d_vs_ref3d']['mean'],
        'visibility_retained_knee_samples':r['knee_visibility_ge_05']['pred3d_vs_ref3d']['n'],
        'all_knee_samples':r['knee_abs_angle_difference_deg']['pred3d_vs_ref3d']['n'],
        'top3_position_joints':sorted(r['joint_position_discrepancy_px'],key=lambda n:r['joint_position_discrepancy_px'][n]['mean'],reverse=True)[:3],
        'low_resolution_position_px':low['position_discrepancy_px']['mean'] if low else None,
        'reference_archive_verified':r['reference']['archive_md5_verified']})
def pg2_greater(s,variant='full',kind='pred3d_vs_ref3d'):
    return knee(index[(s,'PG2',variant,800)],kind)>knee(index[(s,'PG1',variant,800)],kind)
available_heavy=[c for c in comparisons if c['heavy_3d_knee_deg'] is not None]
patterns={'people':people,'paired_people':paired,'condition_count':len(rows),
          'full_3d_knee_pg2_greater_people':[s for s in paired if pg2_greater(s)],
          'projection_gap_pg2_greater_people':[s for s in paired if pg2_greater(s,kind='ref2d_vs_ref3d_projection_difference')],
          'heavy_improved_view_count':sum(c['heavy_3d_knee_deg']<c['full_3d_knee_deg'] for c in available_heavy),
          'heavy_compared_view_count':len(available_heavy),
          'top3_position_joint_view_counts':dict(Counter(n for c in comparisons for n in c['top3_position_joints'])),
          'comparisons':comparisons,'participant_means':[],
          'limits':['Each person is counted once in participant means; frames are repeated observations',
                    'Camera labels are two fixed cameras, not controlled angle-only perturbations',
                    'Larger hip/shoulder discrepancies may include landmark-definition differences',
                    'No annotated visibility/turn labels; no causal attribution to occlusion or rotation',
                    'No training, clinical accuracy cutoff, speed or GDI validation']}
for s in paired:
    entry={'subject':s}
    for variant in ['full','heavy']:
        views=[index[(s,c,variant,800)] for c in ['PG1','PG2'] if (s,c,variant,800) in index]
        if len(views)==2:entry[variant+'_3d_knee_deg']=mean([knee(v) for v in views])
    patterns['participant_means'].append(entry)
patterns['both_views_in_frame_diagnostic']=[]
for s in paired:
    data={}
    for camera in ['PG1','PG2']:
        with np.load(OUT/f'subject-{s:02d}/full/{camera}_800_comparison.npz') as z:data[camera]=dict(z)
    assert np.array_equal(data['PG1']['frames'],data['PG2']['frames'])
    mask=np.ones(len(data['PG1']['frames']),dtype=bool)
    for d in data.values():
        ankle=d['reference_xy'][:,[NAMES.index('LAJC'),NAMES.index('RAJC')]]
        mask &= np.isfinite(ankle).all(axis=(1,2)) & ((ankle[...,0]>=0)&(ankle[...,0]<800)&(ankle[...,1]>=0)&(ankle[...,1]<600)).all(axis=1)
        mask &= np.isfinite(d['reference_3d_bend_deg']).all(axis=1)
    diagnostic={'subject':s,'total_frames':len(mask),'kept_frames':int(mask.sum()),'cameras':{}}
    for camera,d in data.items():
        error=np.abs(d['predicted_3d_bend_deg'][mask]-d['reference_3d_bend_deg'][mask])
        projection=np.abs(d['reference_2d_bend_deg'][mask]-d['reference_3d_bend_deg'][mask])
        diagnostic['cameras'][camera]={'full_3d_knee_deg':float(np.nanmean(error)) if np.isfinite(error).any() else None,
                                       'valid_knee_observations':int(np.isfinite(error).sum()),
                                       'reference_projection_gap_deg':float(np.nanmean(projection)) if np.isfinite(projection).any() else None}
    patterns['both_views_in_frame_diagnostic'].append(diagnostic)
patterns['boundary_diagnostic_limits']='Same frames in both views, both reference ankles inside image and both reference 3D knees valid; secondary diagnostic only. Self-occlusion, joint-definition and synchronization uncertainty remain; primary whole-interval results unchanged.'
args.patterns.write_text(json.dumps(patterns,indent=2)+'\n',encoding='utf-8')

stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M')
new=sum(not r['reused_pilot'] for r in rows)
section=f'''<!-- expansion-20261029-start -->
<h2>추가 검증: {len(people)}명 · {len(base)}개 영상 · {len(rows)}조건</h2>
<p>{stamp} KST 갱신. 참여자 {', '.join(map(str,people))}. 기존 1번의 8조건은 재사용·재검산했고, 추가 참여자에서 {new}조건을 실행했다. 제공된 보행 구간 전체와 30프레임 준비 구간, 같은 모델·설정·평가식을 사용했다. 모델에는 각 카메라 영상만 입력했다. 다른 참여자의 회전 구간에 1번의 프레임 번호를 적용하지 않았다.</p>
<div class="note">원자료 확인: 영상은 제공처 체크섬을 검증했다. 기준 좌표를 담은 TAR 전체 검사가 끝나지 않은 행은 ‘잠정’이다. 현재 수치는 연구실 영상에서 계산한 세 점의 기하학적 굽힘각이며 임상적 관절 각도나 보행 능력 점수가 아니다.</div>
<h3>사람과 카메라별 비교</h3><table><tr><th>참여자·카메라</th><th>기준 묶음</th><th>Full 위치 차이(px)</th><th>Full 3D 무릎 차이(°)</th><th>Heavy 3D 무릎 차이(°)</th><th>정밀 좌표의 2D↔3D 각도 차이(°)</th></tr>
'''
for c in comparisons:
    heavy=f"{c['heavy_3d_knee_deg']:.2f}" if c['heavy_3d_knee_deg'] is not None else '대기'
    status='검사 완료' if c['reference_archive_verified'] else '잠정'
    section+=f"<tr><td>{c['subject']} · {c['camera']}</td><td>{status}</td><td>{c['full_position_px']:.2f}</td><td>{c['full_3d_knee_deg']:.2f}</td><td>{heavy}</td><td>{c['reference_2d_3d_projection_gap_deg']:.2f}</td></tr>\n"
section+='</table><p>입력 800×600 기준. 위치는 12개 대응 관절, 각도는 양쪽 무릎의 평균 절대 차이다. 마지막 열은 AI 추정 없이 정밀 좌표만으로 계산한 투영 차이다. 표의 카메라별 행이나 프레임을 독립적인 사람 수로 세지 않는다.</p>'
section+=f'''<h3>반복 여부를 확인한 관찰</h3><ul>
<li>두 시점이 모두 준비된 {len(paired)}명 중 {len(patterns['full_3d_knee_pg2_greater_people'])}명에서 Full의 3D 무릎 각도 차이가 PG2에서 더 컸다. 정밀 좌표의 투영 차이도 {len(patterns['projection_gap_pg2_greater_people'])}명에서 PG2가 더 컸다. 이는 같은 두 카메라 구성에서의 반복이며, 촬영 각도만의 원인 실험은 아니다.</li>
<li>Heavy는 비교 가능한 {len(available_heavy)}개 영상 중 {patterns['heavy_improved_view_count']}개에서 3D 무릎 각도 차이가 줄었다. 그 이득이 모든 사람·시점에 동일하지는 않다.</li>
<li>고관절·어깨가 위치 차이 상위에 반복해서 나타나는지 아래 표로 확인한다. 기준 관절 중심과 AI 관절점의 정의 차이를 먼저 점검해야 하며, 해당 차이를 모두 AI 오류로 간주하지 않는다.</li>
<li>모델의 visibility 값이 0.5 이상인 무릎만 남긴 결과도 따로 비교했다. 이 값은 실제 가림 정답이 아니며, 선택한 관측 수와 남은 각도 차이를 함께 봐야 한다.</li></ul>
<table><tr><th>참여자·카메라</th><th>위치 차이가 큰 관절 3곳</th><th>visibility≥0.5 각도 차이(°)</th><th>남은 무릎 관측</th><th>발목 기준점이 화면 밖인 프레임</th><th>확인 화면</th></tr>'''
labels={'LHIP':'왼 고관절','RHIP':'오른 고관절','LSJC':'왼 어깨','RSJC':'오른 어깨','LEJC':'왼 팔꿈치','REJC':'오른 팔꿈치','LWJC':'왼 손목','RWJC':'오른 손목','LKJC':'왼 무릎','RKJC':'오른 무릎','LAJC':'왼 발목','RAJC':'오른 발목'}
for c in comparisons:
    image=OUT/f"subject-{c['subject']:02d}/full/{c['camera']}_review.jpg"
    section+=f"<tr><td>{c['subject']} · {c['camera']}</td><td>{', '.join(labels[n] for n in c['top3_position_joints'])}</td><td>{c['full_3d_knee_visibility_ge_05_deg']:.2f}</td><td>{c['visibility_retained_knee_samples']}/{c['all_knee_samples']}</td><td>{c['reference_ankle_outside_image_frame_count']}/{c['frames']}</td><td><a href='{image.as_uri()}'>정지 화면</a></td></tr>"
section+='</table><p>정지 화면은 시작·중간·끝과 위치 차이가 가장 큰 프레임이다. 회전·가림의 시작과 끝을 표시한 정답은 아직 없으므로 오차의 원인으로 확정하지 않았다. 해상도 감소는 영상 전체 축소 시험이며 실제 원거리 촬영과 다르다. 보행 속도·몸통 회전의 정확성·GDI는 이번 비교 범위 밖이다.</p>'
section+='<p><b>화면 경계 문제도 별도로 확인했다.</b> 유효한 정밀 발목 좌표를 원본 800×600 화면에 투영해, 양쪽 중 하나라도 화면 밖인 프레임을 세었다. 이는 영상 구도의 제한을 확인하는 항목이며 모델의 신뢰도나 사람 몸에 가려진 정도를 판정한 값은 아니다. 위 오차 수치는 해당 구간을 빼거나 평가 기준을 바꾸지 않은 값이다. 사람이 검출됐다고 해서 발목까지 관찰 가능하다는 뜻은 아니다.</p>'
section+='<h3>화면 잘림이 없는 공통 프레임만 비교하면?</h3><p>두 시점 모두 양 발목의 기준점이 화면 안에 있고 양 무릎 기준 각도가 유효한 같은 프레임만 선택한 보조 집계다. 기존 전체 구간 결과를 대체하지 않는다. 몸 자체에 가려지는 경우나 관절 정의·동기화 오차는 제거하지 못한다.</p><table><tr><th>참여자</th><th>선택 프레임 / 전체</th><th>PG1 Full 3D 무릎 차이(°)</th><th>PG2 Full 3D 무릎 차이(°)</th></tr>'
for d in patterns['both_views_in_frame_diagnostic']:
    values=[d['cameras'][c]['full_3d_knee_deg'] for c in ['PG1','PG2']]
    text_values=[f'{v:.2f}' if v is not None else '계산 불가' for v in values]
    section+=f"<tr><td>{d['subject']}</td><td>{d['kept_frames']}/{d['total_frames']}</td><td>{text_values[0]}</td><td>{text_values[1]}</td></tr>"
section+='</table>'
section+='<p>원본과 실험 결과는 D드라이브에 보관하며 새 GIF·MP4는 만들지 않았다. '
for path,label in [(args.results,'전체 조건 수치'),(args.patterns,'사람·시점별 집계'),(args.verification,'저장 결과 재검산')]:section+=f'<a href="{path.as_uri()}">{label}</a> · '
section+='</p><hr><h2>기존 1번 참여자 실험</h2>\n<!-- expansion-20261029-end -->'
report=PILOT/'experiment-results.html';html=report.read_text(encoding='utf-8')
start=html.index('<!-- expansion-20261029-start -->');end=html.index('<!-- expansion-20261029-end -->')+len('<!-- expansion-20261029-end -->')
html=html[:start]+section+html[end:]
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key not in ['href','src']:continue
            u=urlparse(value)
            if u.scheme in ['http','https']:continue
            target=Path(unquote(u.path).lstrip('/')) if u.scheme=='file' else PILOT/unquote(value)
            assert target.exists(),str(target)
Links().feed(html)
report.write_text(html,encoding='utf-8')
print(json.dumps(patterns,indent=2))
