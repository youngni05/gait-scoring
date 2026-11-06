"""Independent angle calculation and readable evidence in the existing report."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from run_ready_subjects import ROOT,OUT as BASE,np,loadmat,digest,NAMES
OUT=ROOT/'experiments'/'hip-definition'
a=json.loads((OUT/'analysis.json').read_text());b=json.loads((OUT/'reference-origin-audit.json').read_text())
assert a['plan']['source_results_sha256']==digest(BASE/'results.json')
assert b['plan']['source_analysis_sha256']==digest(OUT/'analysis.json')
assert len(a['results'])==len(b['results'])==120
def atan_bend(x):
    out=[]
    for h,k,f in [(6,8,10),(7,9,11)]:
        u=x[:,h]-x[:,k];v=x[:,f]-x[:,k]
        u=u.astype(np.float64);v=v.astype(np.float64)
        cross=np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0]) if x.shape[-1]==2 else np.linalg.norm(np.cross(u,v),axis=-1)
        angle=180-np.degrees(np.arctan2(cross,np.sum(u*v,axis=-1)))
        good=(np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)>1e-10)
        out.append(np.where(good,angle,np.nan))
    return np.stack(out,axis=-1)
def mean(x):return float(np.nanmean(x))
def assert_close(x,y):assert abs(x-y)<1e-6,(x,y)
checks=[]
for subject in a['plan']['people']:
    m=loadmat(ROOT/'reference'/f'F_v3d_Subject_{subject}.mat',simplify_cells=True)[f'Subject_{subject}_F']['move']
    assert digest(ROOT/'reference'/f'F_v3d_Subject_{subject}.mat')==next(r['file_sha256'] for r in a['metadata'] if r['subject']==subject)
    names=[n if isinstance(n,str) else '' for n in m['markerName']]
    for ar in [r for r in a['results'] if r['subject']==subject]:
        camera,variant=ar['camera'],ar['variant'];p=BASE/f'subject-{subject:02d}'/variant/f'{camera}_800_comparison.npz'
        assert digest(p)==ar['comparison_sha256']
        with np.load(p) as z:d=dict(z)
        ref=d['reference_xy'];pred=d['predicted_xy'];e=pred[:,[6,7]]-ref[:,[6,7]]
        mse=np.nanmean(np.sum(e**2,axis=-1));side_counts=np.isfinite(e).all(axis=-1).sum(axis=0)
        variance=np.nansum(np.nanvar(e,axis=0),axis=-1)
        variance=float((variance*side_counts).sum()/side_counts.sum())
        assert_close((mse-variance)/mse,ar['bias_energy_fraction'])
        truth=atan_bend(ref);pred_angle=atan_bend(pred)
        only=ref.copy();only[:,[6,7]]=pred[:,[6,7]]
        oracle=pred.copy();oracle[:,[6,7]]=ref[:,[6,7]]
        only_angle=atan_bend(only);oracle_angle=atan_bend(oracle)
        common=np.isfinite(truth)&np.isfinite(pred_angle)&np.isfinite(only_angle)&np.isfinite(oracle_angle)
        for field,value in [('full_2d_knee_mae_deg',pred_angle),('hip_only_2d_knee_mae_deg',only_angle),
                            ('reference_hip_substitution_2d_knee_mae_deg',oracle_angle)]:
            assert_close(ar[field],mean(np.abs(value-truth)[common]))
        # Reconstruct the approximate CODA axes by an orthogonal projection,
        # instead of the cross-product order used by the audit script.
        idx=(d['frames']-1)*4;pi=[names.index(n) for n in ['LFWT','RFWT','LBWT','RBWT']]
        markers=m['markerLocation'][idx][:,pi]
        valid=~m['markerGaps'][idx][:,pi].astype(bool).any(axis=-1)
        front=(markers[:,0]+markers[:,1])/2;back=(markers[:,2]+markers[:,3])/2
        lateral=markers[:,1]-markers[:,0];lateral/=np.linalg.norm(lateral,axis=-1,keepdims=True)
        forward=front-back;forward-=np.sum(forward*lateral,axis=-1,keepdims=True)*lateral
        forward/=np.linalg.norm(forward,axis=-1,keepdims=True);up=np.cross(lateral,forward)
        width=np.median(np.linalg.norm(markers[:,0]-markers[:,1],axis=-1)[valid])
        new3=d['reference_3d'].copy()
        for side,sign in [(6,-1),(7,1)]:new3[:,side]=front+width*(.36*sign*lateral-.19*forward-.3*up)
        new3[~valid,6:8]=np.nan
        old_angle=atan_bend(d['reference_3d']);new_angle=atan_bend(new3);model_angle=d['predicted_3d_bend_deg']
        common3=np.isfinite(old_angle)&np.isfinite(new_angle)&np.isfinite(model_angle)
        br=next(r for r in b['results'] if (r['subject'],r['camera'],r['variant'])==(subject,camera,variant))
        assert int(common3.sum())==br['angle_samples']
        assert_close(mean(abs(model_angle-old_angle)[common3]),br['provided_reference_3d_mae_common_deg'])
        assert_close(mean(abs(model_angle-new_angle)[common3]),br['document_coda_reference_3d_mae_common_deg'])
        assert_close(mean(abs(new_angle-old_angle)[common3]),br['reference_3d_angle_change_deg'])
        checks.append({'subject':subject,'camera':camera,'variant':variant,'passed':True})
    print('Independently checked',subject,flush=True)
verification={'analysis_sha256':digest(OUT/'analysis.json'),'origin_audit_sha256':digest(OUT/'reference-origin-audit.json'),
 'baseline_results_sha256_unchanged':digest(BASE/'results.json'),'people':30,'conditions_audited':120,
 'independent_methods':['atan2 cross/dot angles versus arccos','orthogonal projection versus cross-product CODA axes',
                        'variance identity versus explicit bias subtraction','source and cache hashes'],
 'checks':checks,'passed':True}
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11})
fig,ax=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
# Example is fixed in advance: first walking frame, subject 1, not a best-fit example.
m=loadmat(ROOT/'reference'/'F_v3d_Subject_1.mat',simplify_cells=True)['Subject_1_F']['move'];names=list(m['markerName'])
with np.load(BASE/'subject-01'/'full'/'PG1_800_comparison.npz') as z:
    example_frames=(z['frames']-1)*4;f=int(example_frames[0])
pi=[names.index(n) for n in ['LFWT','RFWT','LBWT','RBWT']];marks=m['markerLocation'][f,pi]
A=marks[:2].mean(axis=0);P=marks[2:].mean(axis=0);W=np.linalg.norm(marks[1]-marks[0])
example_good=~m['markerGaps'][example_frames][:,pi].astype(bool).any(axis=-1)
example_markers=m['markerLocation'][example_frames][:,pi]
example_width=np.median(np.linalg.norm(example_markers[:,0]-example_markers[:,1],axis=-1)[example_good])
x=(marks[1]-marks[0])/W;up=np.cross(x,A-P);up/=np.linalg.norm(up);y=np.cross(up,x)
def top(points):
    delta=np.asarray(points)-A
    return np.stack([delta@x,delta@y],axis=-1)/W
q=top(marks);hip=top(m['markerLocation'][f,[names.index('LHIP'),names.index('RHIP')]])
origin=top(m['jointsAffine_v3d'][f,list(m['segmentName']).index('RPV'),:3,3])
ax[0].plot(q[[0,1,3,2,0],0],q[[0,1,3,2,0],1],color='#bdc4cc',lw=1.5)
ax[0].scatter(q[:,0],q[:,1],s=35,color='#7c8795',label='골반 앞·뒤 마커')
ax[0].scatter(0,0,s=110,marker='+',color='black',label='앞 마커 중점: 문서의 CODA 원점')
ax[0].scatter(*origin,s=65,marker='s',color='#9467bd',label='MAT에 저장된 골반 원점')
ax[0].scatter(hip[:,0],hip[:,1],s=95,marker='x',color='#07867e',label='제공된 고관절 좌표')
ax[0].scatter(np.array([-.36,.36])*example_width/W,np.array([-.19,-.19])*example_width/W,s=65,marker='D',color='#e38d20',label='문서식 근사 재구성')
ax[0].set(title='실제 좌표에서 본 골반: 위에서 내려다본 모습\n1번 참여자, 첫 보행 프레임',xlabel='좌우 위치 / 골반 너비',ylabel='앞뒤 위치 / 골반 너비 (위쪽=앞)')
ax[0].set_aspect('equal');ax[0].set_xlim(-.75,.75);ax[0].set_ylim(-.95,.25);ax[0].legend(loc='lower left',fontsize=8)
s=[b['summary'][v]['equal_person_view_means'] for v in ['full','heavy']];pos=np.arange(2)
for offset,key,color,label in [(-.19,'provided_reference_3d_mae_common_deg','#07867e','제공 좌표와 비교'),
                               (.19,'document_coda_reference_3d_mae_common_deg','#e38d20','문서식 근사 좌표와 비교')]:
    values=[r[key] for r in s];bars=ax[1].bar(pos+offset,values,.36,color=color,label=label)
    ax[1].bar_label(bars,fmt='%.2f°',padding=4)
ax[1].set_xticks(pos,['Full','Heavy']);ax[1].set_ylim(0,18)
ax[1].set(title='같은 예측도 비교 기준에 따라 값이 달라진다',ylabel='3D 무릎 각도의 평균 절대 차이 (°)')
ax[1].legend(loc='upper right',fontsize=9)
fig.supxlabel('30명 · 각 2시점 · 사람별 동일 가중치. 근사 재구성은 정답 미확정이며, 모델 성능 개선 결과가 아니다.',fontsize=10)
fig.savefig(OUT/'definition-sensitivity.png',dpi=150,bbox_inches='tight',pad_inches=.18);plt.close(fig)

report=Path(__file__).resolve().parents[2]/'20261028-movi-totalcapture-acquisition'/'experiment-results.html'
full=a['summary']['full']['equal_person_view_means'];heavy=a['summary']['heavy']['equal_person_view_means']
table=''
for label,key in [('고관절 화면 위치 차이(px)','hip_distance_common_px'),('화면상 2D 무릎 각도 차이(°)','reference_2d_mae_common_deg'),('3D 무릎 각도 차이(°)','reference_3d_mae_common_deg')]:
    if key=='hip_distance_common_px':oldkey='provided_'+key;newkey='document_coda_'+key
    else:oldkey='provided_'+key;newkey='document_coda_'+key
    table+=f'<tr><td>{label}</td>'+''.join(f'<td>{r[k]:.2f}</td>' for r in s for k in [oldkey,newkey])+'</tr>'
section=f'''<!-- hip-definition-20261029-start -->
<section id="hip-definition-diagnostic"><p class="lead"><b>고관절 정의 확인: 기준점의 생성 방식부터 맞춰야 한다.</b></p>
<p>30명×2시점에서 기존 Full·Heavy 800픽셀 예측 120조건을 재사용했다. 새 모델 실행·학습·오차 맞춤 보정 없이 정의와 기하학적 영향을 진단했다. 아래 분석은 기존 240조건의 정답·예측·성능 기록을 대체하지 않는다.</p>
<p><b>MoVi</b>는 골반 마커와 Bell–Brand 회귀식으로 관절 중심을 추정한다. MAT의 LHIP/RHIP가 이 고관절이며 LHJC/RHJC는 손 관절이다. 기존 대응은 올바르다. <b>MediaPipe</b>는 사람의 영상 관절 주석과 GHUM 인체 모델에서 학습한 23/24번 고관절을 예측한다. 공개 자료로 두 방식의 정확한 해부학적 간격을 확정하지 못했다.
<a href="{a['plan']['sources']['movi']}">MoVi 논문 §2.7.1</a> · <a href="{a['plan']['sources']['model_card']}">MediaPipe 모델 카드</a></p>
<p><b>현재 관측:</b> Full의 고관절 위치 차이는 평균 {full['hip_mean_distance_px']:.1f}px이고, 60개 영상의 고관절 평균 잔차가 모두 화면 아래쪽이었다(평균 {full['hip_mean_delta_y_px']:.1f}px). 어깨·무릎·발목이 같은 만큼 아래로 밀린 것은 아니었다. 양 고관절이 함께 이동하는 성분이 좌우 폭·방향 차이보다 컸다. visibility≥0.5에서도 고관절 관측은 모두 남아 같은 차이를 보였다.</p>
<p><b>고관절만 바꿔 넣는 진단:</b> 무릎·발목 예측은 유지하고 고관절만 제공된 기준 좌표로 대체하면 화면상 2D 각도 차이는 Full {full['full_2d_knee_mae_deg']:.2f}→{full['reference_hip_substitution_2d_knee_mae_deg']:.2f}°, Heavy {heavy['full_2d_knee_mae_deg']:.2f}→{heavy['reference_hip_substitution_2d_knee_mae_deg']:.2f}°였다. Full은 60영상 모두 감소했고, 양 시점의 기준 발목이 화면 안인 공통 프레임에서도 8.03→4.13°였다. 정답을 미리 넣은 계산이므로 실제 사용 가능한 개선 성능이 아니다.</p>
<p><b>기준 생성 과정의 추가 확인점:</b> 공개 CODA 문서는 앞쪽 골반 마커(ASIS) 중점을 원점으로 삼는다. MAT의 PELVIS 점은 앞·뒤 네 마커의 평균과 일치하며, 저장된 골반 변환 원점도 대체로 그 부근이다. 이 원점은 문서상 앞 마커 중점과 평균 골반 너비의 약 34% 떨어져 있었다. 28명은 저장된 골반 변환 원점에 Bell 계수를 적용한 형식과 수치적으로 일치했고, 2·31번은 그 관계에도 공통 이동 차이가 있었다. 따라서 저장된 골반 변환을 문서의 CODA 좌표계로 그대로 해석하는 것은 확인이 필요하다.
<a href="{b['plan']['sources']['coda']}">CODA 원점·축 정의</a> · <a href="{b['plan']['sources']['hip_formula']}">Bell–Brand 공식</a></p>
<div class="note"><b>다음 표는 참고 기준의 정의를 달리했을 때의 민감도이며, 교정된 정답이나 모델 개선 성적이 아니다.</b> 앞·뒤 골반 마커로 CODA 축을 구성하고 관측된 앞 마커 간 거리의 사람별 중앙값으로 골반 너비를 근사했다. 원래 Visual3D의 개인별 설정값·마커 반지름·내보내기 변환은 복원하지 못했다. 정적 lcData가 0인 49번을 포함해 모두 같은 관측 거리 절차를 썼다. 예측에 맞춰 계수·시간을 조정하지 않았다.</div>
<table><tr><th>같은 관측으로 짝지어 비교</th><th>Full 제공 좌표</th><th>Full 문서식 근사</th><th>Heavy 제공 좌표</th><th>Heavy 문서식 근사</th></tr>{table}</table>
<p>근사 구성으로 고관절 위치 차이가 줄어든 영상은 두 모델 각각 59/60개, Full의 3D 각도 차이가 줄어든 영상은 54/60개였다. 기준 무릎 각도 자체는 평균 9.72° 달라졌다. 전체 골격의 좌표 원점을 바꾸면 각도는 그대로지만, 고관절 생성의 기준점을 바꾸고 무릎·발목은 유지하면 허벅지 방향이 달라지기 때문이다.</p>
<img src="{(OUT/'definition-sensitivity.png').as_uri()}" alt="실제 골반 마커·고관절의 앞뒤 위치 차이와 참고 정의에 따른 각도 비교값 변화">
<p><b>판단:</b> 고관절 위치 차이는 무시하기 어렵고, 비교 기준 생성법이 수치를 크게 바꾼다. 그러나 일정한 편향이나 오차 감소량을 ‘해부학적 정의 차이가 설명한 비율’로 단정할 수 없다. 더 작은 차이를 보이는 근사 구성이 해부학적으로 더 정확하다는 증거도 아직 없다. 모델 순위·좌표 보정법을 정하기 전에 원래 골반 좌표계와 고관절 생성 설정을 확인하는 것이 우선이다. 현재 Heavy 우위도 제공된 기준 좌표에 대한 결과로 해석해야 한다.</p>
<details><summary>계산·검증 범위와 근거 파일</summary><p>사람별·시점별 계산 뒤 30명에 동일 가중치를 부여했다. 원점 감사는 유효한 네 골반 마커가 있는 공통 관측에서 비교했고, 한 모델당 좌우·두 시점을 합친 각도 관측 32,506개를 사용했다. 이 관측들을 독립 사람 수로 세지 않았다. Full 고관절 제곱오차의 평균 {100*full['bias_energy_fraction']:.1f}%는 영상 내 일정한 평균 잔차 성분이었으나 그 원인은 분리되지 않는다. 두 고관절의 함께 이동하는 성분은 제곱오차의 평균 {100*full['midpoint_energy_fraction']:.1f}%였다. 각도는 세 점의 기하학적 굽힘각이며 임상 관절좌표계 각도가 아니다. atan2 방식으로 독립 재계산했고 CODA 축도 다른 수식으로 재구성해 120조건을 검산했다.</p>
<p><a href="{(OUT/'analysis.json').as_uri()}">사람별 진단 수치·출처</a> · <a href="{(OUT/'reference-origin-audit.json').as_uri()}">골반 원점·정의 민감도</a> · <a href="{(OUT/'verification.json').as_uri()}">독립 검산</a></p></details></section>
<!-- hip-definition-20261029-end -->'''
html=report.read_text(encoding='utf-8')
start='<!-- hip-definition-20261029-start -->';end='<!-- hip-definition-20261029-end -->'
if start in html:
    left=html.index(start);right=html.index(end,left)+len(end);html=html[:left]+section+html[right:]
else:html=html.replace('<!-- expansion-20261029-start -->',section+'\n<!-- expansion-20261029-start -->',1)
assert html.count(start)==1
report.write_text(html,encoding='utf-8')
print('Verified 120 conditions; wrote figure and existing report.')
