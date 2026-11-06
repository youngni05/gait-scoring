"""Summarize the verified comparison; reference depth is explanation only."""
import json,io,tarfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import experiment as e

def main():
    result=e.read(e.H/'comparison.json');verify=e.read(e.H/'verification.json');assert verify['status']=='passed' and verify['comparison_sha256']==e.sha(e.H/'comparison.json')
    archive=e.ROOT/'raw/Camera Parameters.tar';diagnostics=[]
    with tarfile.open(archive) as t:
        extrinsic={cam:dict(np.load(io.BytesIO(t.extractfile(f'Calib/Extrinsics_{cam}.npz').read()))) for cam in ['PG1','PG2']}
    for q in result['calibrations']:
        sid=q['subject'];cam=q['camera'];meta=loadmat(e.ROOT/'reference'/f'F_v3d_Subject_{sid}.mat',simplify_cells=True)[f'Subject_{sid}_F']['move'];names=[x if isinstance(x,str) else '' for x in meta['markerName']];indices=[names.index(x) for x in ['LFWT','LBWT','RFWT','RBWT']];pose=meta['markerLocation'][:,indices].mean(1);gaps=meta['markerGaps'][:,indices].astype(bool).any(1)
        standindices=(np.arange(31,91)-1)*4;window=next(x for x in e.primary() if x['key']==q['key']);walkindices=(np.arange(window['first_frame'],window['last_frame']+1)-1)*4
        ex=extrinsic[cam];camera=pose@ex['rotationMatrix']+ex['translationVector'];ok=not gaps[np.r_[standindices,walkindices]].any();standz=float(np.median(camera[standindices,2])/1000) if ok else None;walkz=float(np.median(camera[walkindices,2])/1000) if ok else None
        diagnostics.append({'key':q['key'],'subject':sid,'camera':cam,'reference_only_do_not_feed_estimator':True,'reference_markers_available':ok,'standing_camera_depth_m':standz,'walking_camera_depth_m':walkz,'walking_minus_standing_depth_m':walkz-standz if ok else None,'standing_divided_by_walking_depth':standz/walkz if ok else None,'scale_ratio':q['scale_ratio'],'standing_span_cv':q['standing_span_cv']})
    e.save(e.H/'diagnostics.json',{'status':'posthoc_explanation_only_not_causal_proof','camera_archive_sha256':e.sha(archive),'rows':diagnostics})
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for i,cam in enumerate(['PG1','PG2']):
        for j,(metric,label,factor) in enumerate([('speed','Speed error (m/s)',1),('step_length','Step length error (cm)',100),('stride_length','Stride length error (cm)',100)]):
            ax=axes[i,j];rows=[p for p in result['pairs'] if p['camera']==cam and p['metric']==metric and p['paired_available']]
            for q in rows:
                a=abs(q['baseline']-q['reference'])*factor;b=abs(q['standing']-q['reference'])*factor;ax.plot([0,1],[a,b],color='#178178' if b<a else '#c9653b',alpha=.65,marker='o',lw=1,ms=3)
            means=[np.mean([abs(q[k]-q['reference'])*factor for q in rows]) for k in ['baseline','standing']] if rows else [np.nan,np.nan]
            ax.set_title(f'{cam}: {label}, n={len(rows)}',fontsize=10)
            if not rows:
                ax.text(.5,.5,'No paired measurements',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
            ax.plot([0,1],means,color='black',lw=2.5,marker='D',label='Mean error');ax.set_xticks([0,1],['Walking height','Standing height']);upper=max(abs(q[k]-q['reference'])*factor for q in rows for k in ['baseline','standing']);ax.set_ylim(0,max(upper*1.15,.01));ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Same walking coordinates, events and windows; only height scale changes\nYoung-adult laboratory videos. Green: lower error; orange: higher error.',fontsize=11);fig.tight_layout(rect=(0,0,1,.92));fig.savefig(e.H/'paired-errors.png',dpi=180);fig.savefig(e.H/'paired-errors.pdf');plt.close(fig)
    lines=['서 있는 자세 기준 키 보정과 기존 보행 기준 키 보정의 비교','확인일: 2026-11-05','','실제 영상의 같은 14명·28개 구도별 보행 구간에서 거리 환산 비율만 비교했다. 보행 좌표, 접지 시점, 품질 판정, 정답과 구간은 바꾸지 않았다. 주분석은 측면에 가까운 PG1이며 PG2는 비스듬한 구도의 대조다. 고령자·실제 휴대폰 검증은 아니다.','','방법: 기존은 보행 구간의 관절점 세로 범위 중앙값으로 키를 나눴다. 새 방식은 동일 촬영의 31~90프레임(2초) 서 있는 구간으로 같은 계산을 하고, 그 비율을 보행 구간에 고정 적용했다. 1~30프레임은 추적 준비다. 서 있는 구간은 전체 28개 영상의 60프레임을 먼저 시각 확인했고, 모든 자료에서 첫 동작 주석 이전임을 확인했다. 새 추출은 기존과 같은 MediaPipe Heavy, 800×600, 30fps, CPU 모델과 설정을 사용했다.','','실제 정수리-발바닥 검출을 새로 도입하지 않았으므로, 관절점 높이와 실제 키의 정의 차이는 여전히 남는다. 서 있는 위치와 걷는 위치의 카메라 거리가 다를 수 있다.','','평균절대오차: 각 항목은 두 방식 모두 계산 가능한 동일 사람만 비교했다. 변화량=새 방식 오차−기존 오차이며, 음수이면 개선이다. 95% 불확실성은 사람 단위 짝지은 재표집 10,000회다.']
    lines[3:3]=['측면 주분석에서 속도 평균오차는 0.0906→0.0793m/s, 한 걸음 길이는 5.53→4.96cm로 감소했다. 다만 두 항목의 짝지은 95% 불확실성은 0을 포함하여 개선을 확정하기 어렵다. 속도가 개선된 사람은 10/14명이지만 4명은 악화했고, 큰 속도 오차의 95백분위도 0.1426→0.1508m/s로 증가했다.','']
    verdicts={'inconclusive':'개선 여부 확정 어려움','reduction_supported':'이 짝지은 표본에서 오차 감소 지지','increase_supported':'이 짝지은 표본에서 오차 증가 지지'}
    for cam in ['PG1','PG2']:
        lines.append('\n'+cam)
        for field,label,factor,unit in [('speed','속도',1,'m/s'),('step_length','한 걸음 길이',100,'cm'),('stride_length','한 보행 주기 길이',100,'cm')]:
            q=result['summary'][cam][field];ci=q['paired_change_CI95']
            if q['paired_people']==0:
                lines.append(f"{label}: 짝지은 비교 측정 불가. 기존만 유효 {q['baseline_available_people']}/14명.");continue
            uncertainty=f"95% [{ci[0]*factor:+.4f}, {ci[1]*factor:+.4f}]" if ci else '인원 부족으로 신뢰구간 제시 불가'
            lines.append(f"{label}: 기존 {q['baseline_MAE']*factor:.4f} → 서 있는 기준 {q['standing_MAE']*factor:.4f} {unit}; 동일 {q['paired_people']}/14명; 기존 방식 유효 {q['baseline_available_people']}명. 오차 변화 {q['MAE_change_standing_minus_baseline']*factor:+.4f} {unit}; {uncertainty}. 개선 {q['improved_people']}명, 악화 {q['worse_people']}명. 판정: {verdicts[q['interpretation']]}.")
    failures=[q['key'] for q in result['calibrations'] if not q['standing_quality']];lines+=['','서 있는 구간의 품질 미충족: '+(', '.join(failures) if failures else '없음. 28개 구도 모두 통과.'),'','원인 해석을 돕는 사후 진단: 광학식 골반 마커와 제공자 카메라 보정으로 서 있을 때/걸을 때의 카메라 깊이를 계산했다. 이 값은 결과 설명에만 썼고 새 보정의 입력·조건 선택·결과 조정에 사용하지 않았다. 카메라 깊이 변화는 가능성 있는 원인을 보여주며 인과관계를 확정하지 않는다.']
    for cam in ['PG1','PG2']:
        ds=[q for q in diagnostics if q['camera']==cam and q['reference_markers_available']];depth=np.array([q['walking_minus_standing_depth_m'] for q in ds]);ratios=[q['scale_ratio'] for q in result['calibrations'] if q['camera']==cam and q['standing_quality']]
        depthtext=f"중앙값 {np.median(depth):+.3f}m, 범위 {depth.min():+.3f}~{depth.max():+.3f}m" if len(depth) else '기준 마커 결측으로 측정 불가'
        scaletext=f"중앙값 {np.median(ratios):.4f}, 범위 {min(ratios):.4f}~{max(ratios):.4f}" if ratios else '측정 불가'
        lines.append(f"{cam}: 기준 깊이 진단 {len(ds)}명; 걷기−서기 카메라 깊이 {depthtext}. 새/기존 환산 비율 {scaletext}.")
    lines+=['','이 비교에서 시간 지표는 환산 비율의 영향을 받지 않고, 좌우 비대칭 비율도 양쪽에 같은 상수를 곱하면 변하지 않는다. 따라서 이 결과를 걸음 시간·좌우 차이 정확도 개선으로 확대하지 않는다.','','독립 검산: 새 좌표 28개와 실행 기록 28개의 SHA256, 원본 영상/모델/이전 결과의 해시, 서 있는/보행 프레임과 실제 키, 원본 좌표에서 재계산한 속도·길이, 독립 마커 기준, 동일 사람과 짝지은 오차·불확실성을 확인했다. 수치 허용오차 1e-10을 유지했다. 신경망 학습·GPU·다운로드·외부 전송은 없다.','','파일: comparison.json, paired-measurements.csv, verification.json, diagnostics.json, protocol.json, paired-errors.png/pdf','새 좌표 및 실행 기록: '+str(e.B)]
    (e.H/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    e.save(e.H/'STATE.json',{'status':'paired_comparison_and_independent_verification_completed','verification':'passed','people':14,'walking_windows_views':28,'new_CPU_inference_frames':2520,'training':False,'limitations':['initial standing location differs from walking','landmark span is not true head-top height','18..33-year-old laboratory population','scale-only comparison, not re-detected contacts'],'comparison_sha256':e.sha(e.H/'comparison.json'),'report_sha256':e.sha(e.H/'RESULTS.txt')})
    files=sorted(p for p in e.H.iterdir() if p.is_file() and p.name!='artifact-manifest.json');e.save(e.H/'artifact-manifest.json',{'files':{p.name:{'sha256':e.sha(p),'bytes':p.stat().st_size} for p in files}})
    print(json.dumps({'report_written':True,'artifact_files':len(files),'standing_failures':failures}))

if __name__=='__main__':main()
