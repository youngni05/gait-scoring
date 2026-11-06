"""Factual report assembled only after complete replay verification."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import prepare as p
H=p.H;B=p.B
NAMES={'A':'속도만','A2':'속도 변화 추가','B':'기존 보행 특징','C':'관절 시계열 CNN + 속도','D':'관절 CNN + 보행 특징','R':'고정 시간 필터 + 특징','E':'학습 내부 예측 결합'}

def main():
    r=p.read(H/'results.json');v=p.read(H/'verification.json');diag=p.read(H/'diagnosis.json');assert v['status']=='passed' and p.sha(H/'results.json')==v['results_sha256'];assert p.sha(H/'person-predictions.csv')==v['prediction_table_sha256']
    lines=[]
    for arm in NAMES:
        all_=r['summary']['all'][arm];old=r['summary']['age65plus'][arm];slow=r['summary']['TUG20plus'][arm]
        lines.append(f"{NAMES[arm]}: 전체 {all_['MAE']:.3f}초 / 65세 이상 {old['MAE']:.3f}초 / TUG≥20초 {slow['MAE']:.3f}초; 2초 이내 {all_['within2']:.1%} / 고령층 {old['within2']:.1%}")
    contrast=[]
    for arm in ['C','D','R','E']:
        z=r['paired']['all'][arm+'_minus_A'];b=r['paired']['all'][arm+'_minus_B'];o=r['paired']['age65plus'][arm+'_minus_A']
        contrast.append(f"{NAMES[arm]}: 속도 대비 {z['delta_MAE']:+.3f}초(95% {z['CI95'][0]:+.3f}~{z['CI95'][1]:+.3f}), 기존 특징 대비 {b['delta_MAE']:+.3f}초({b['CI95'][0]:+.3f}~{b['CI95'][1]:+.3f}); 고령층 속도 대비 {o['delta_MAE']:+.3f}초({o['CI95'][0]:+.3f}~{o['CI95'][1]:+.3f})")
    shuffle=[]
    for arm in ['C','D']:
        z=r['paired']['all'][arm+'_shuffled_minus_'+arm];shuffle.append(f"{NAMES[arm]}: 시간 순서 섞기 후 MAE 변화 {z['delta_MAE']:+.3f}초({z['CI95'][0]:+.3f}~{z['CI95'][1]:+.3f})")
    anchors={arm:[['raw','residual6overv'][r['selections'][i]['candidates'][arm]['candidate']%2] for i in range(5)] for arm in ['C','D']}
    maximum=max(diag['slow9_profiles'],key=lambda item:item['TUG_seconds'])
    ablation_text='관절 제거 대조 학습은 아직 검산 결과가 없다.'
    ablation=None
    if (H/'ablation-verification.json').exists():
        av=p.read(H/'ablation-verification.json');ablation=p.read(H/'ablation-results.json');assert av['status']=='passed' and p.sha(H/'ablation-results.json')==av['results_sha256']
        statements=[]
        for arm in ['C0','D0']:
            a=ablation['summary']['all'][arm];o=ablation['summary']['age65plus'][arm];s=ablation['summary']['TUG20plus'][arm]
            statements.append(f"{arm} 관절 시계열 제거 대조: 전체 {a['MAE']:.3f}초 / 고령층 {o['MAE']:.3f}초 / TUG≥20초 {s['MAE']:.3f}초")
        for group,label in [('all','전체'),('age65plus','65세 이상')]:
            z=ablation['paired'][group]['D_minus_D0'];statements.append(f"{label} D−D0: {z['delta_MAE']:+.3f}초(95% {z['CI95'][0]:+.3f}~{z['CI95'][1]:+.3f})")
        ablation_text='\n'.join(statements)+f"\n관절 입력을 모든 사람에서 일정한 0으로 바꾸고 같은 구조·네 설정·초기값·학습 방식으로 다시 학습했다. 0은 측정값 대체가 아니라 의도적인 반사실 대조군이다. 추가 내부120개·최종10개 저장 모델 재추론과 CPU64 검산을 통과했다(최대 차이 {av['max_TCN_CPU64_difference']:.3g}초). 첫 결과를 보고 설계한 원인 진단으로, 별도 외부 검증이 아니다.\n전체에서 관절 정보 추가 효과는 약0.008초로 구간이0을 포함한다. 이번 전체 평균 개선을 관절 움직임 자체의 학습 효과로 설명할 근거는 부족하다. TUG≥20초9명에서는 D가 D0보다 약0.525초 낮지만 D의 평균오차 자체가 약11초이므로 개인 정확도 목표와는 거리가 있다."
    success=' / '.join(r['goal_passing_arms']) or '없음'
    continuation=' / '.join(r['promising_for_four_repeat_sensitivity']) or '없음'
    record=f'''OpenCap 짧은 보행의 TUG 예측 — 관절 움직임 학습 비교 기록 (2026-11-05)

실행 상태
297명의 같은 3초 보행 구간과 같은 다섯 사람 분할로 새 학습·원본 재추론·독립 지표 계산을 마쳤다.
전체·65세 이상에서 개발 목표를 모두 만족한 방법: {success}.
사전에 정한 추가 네 분할 반복의 진행 조건을 만족한 방법: {continuation}.
현재 문서의 표는 기존 repeat0 다섯 분할 결과다. 기존의 다섯 반복 평균 1.75초와 혼합 집계하지 않는다.

모델이 답한 질문
별도로 촬영한 3초 정상 속도 보행의 움직임으로 같은 사람·같은 방문의 실제 임상 TUG 완료 시간(tft_tug)을 예측한다.
자료는 신경근 질환 등을 포함한 OpenCap 코호트 297명이며, 65세 이상 34명, TUG 12초 미만 264명·12~20초 24명·20초 이상 9명이다.
일반 고령자만의 집단이나 휴대폰 한 대의 실제 측정 성능으로 해석하지 않는다. 이번 새 학습의 정답은 TUG 초이며 SPPB·JH-FRAT는 학습하지 않았다.

입력과 대조 조건
원본과 연결한 동일 3초·91시점, 30Hz·SG7/order2를 사용했다. 기존 64개 특징을 모두 재생성했으며 최대 차이 0이다.
원본 297개 파일 SHA256과 정답의 사람·방문 연결을 확인했다. 새 속도 계산과 기존 입력 차이 최대 6.67e-16m/s다.
카메라 유래 다중 시점 3D를 이상적인 측면으로 투영한 좌표다. 실제 단일 휴대폰·MediaPipe로 측정한 좌표가 아니다. 속도 m/s를 얻기 위한 공간 보정을 전제로 한다.
새 시계열은 발가락을 제외한 16개 관절의 골반 중심 상대 좌표 32채널과 골반 이동 2채널이다. 보이는 한쪽 평면만 사용하며 횡방향 깊이·보강 해부학 마커를 넣지 않았다.
나이·성별·질환·다른 검사 점수·TUG 동작·전체 기록 길이는 입력에서 제외했다. 원래 특징의 이용 가능성 표시가 일부 포함되므로 실제 추출기의 차이는 여전히 검증 대상이다.
A/A2/B는 기존 저장 모델을 그대로 재추론했다. C/D는 같은 작은 시간 합성곱 신경망이며 폭 16/32와 두 가지 정답 학습 방식을 내부에서 비교했다.
첫 방식은 TUG를 직접 예측하며, 둘째는 가상의 6m/속도 기준값에 대한 잔차를 학습한다. 6m/속도는 실제 TUG의 보행 시간 측정치가 아니다.
R은 무작위로 고정한 시간 필터 256개의 출력과 기존 특징에 Ridge를 학습한다. E는 B/C/D/R의 학습 내부 교차 예측만으로 음수가 아닌 결합 계수를 학습한다.

학습·선택 기준
기존 사람 분할을 그대로 사용했다. 바깥 평가 사람은 설정 선택·정규화·결측 대체·학습 종료 시점에 사용하지 않았다.
C/D는 각각 4설정×3내부 분할, R은 3설정×3내부 분할이다. CNN은 최대100epoch·개선 없는15epoch 종료·내부 최선 MAE를 사용했다. 최종 학습 epoch는 세 내부 최선 epoch의 중앙값이다.
CPU 한 스레드, 기존 설치 환경을 재사용했다. 외부 전송·새 학교 GPU 작업·새 다운로드는 없다. 조건별 학습 횟수는 다르며 서로 다른 예측 방식의 실용 비교이지 순수한 특징 개수만의 효과 실험은 아니다.
C의 다섯 분할 정답 방식: {anchors['C']}
D의 다섯 분할 정답 방식: {anchors['D']}
각 학습의 초기값은 하나다. 다른 초기값·완전히 새로운 외부 사람에 대한 안정성은 확인하지 못했다.

개발 목표
전체와 65세 이상에서 모두 평균오차≤1초, 개인 오차≤2초 비율≥80%, 속도 기준보다 평균오차≥20% 감소와 두 속도 기준 대비 짝비교 개선을 목표로 정했다.
공학적 개발 목표이며 임상 최소 유의 변화량이나 진단 기준이 아니다. 1초를 넘는 오차는 원인 점검 대상으로 남기며 오류 크기만으로 계산 버그로 확정하지 않는다.

평균 절대오차 및 개인 오차 범위
{chr(10).join(lines)}

같은 사람의 오차 차이
{chr(10).join(contrast)}
음수는 새 방식이 유리하다. 95% 구간은 저장된 사람별 교차검증 예측에 대해 10,000회 재표집한 조건부 구간이다. 전체 재학습·여러 모델 탐색의 불확실성을 포함하지 않는다.
기존부터 반복 사용한 개발 자료다. 새 학습·설정 비교 결과를 독립 외부 검증으로 부르지 않는다. 여러 비교의 다중 검정 보정은 하지 않았다.

시간 순서 진단
{chr(10).join(shuffle)}
동일 학습 모델에 시간 순서를 섞어 입력한 진단이다. 다시 학습한 공정 대조 실험이 아니며 변화량만으로 시간 구조의 임상적 유용성을 증명할 수 없다.

관절 제거 후 같은 방식으로 다시 학습한 추가 원인 진단
{ablation_text}

큰 오차의 원인 점검
평가 사람 중복·잘못 연결된 방문 정답·기존 특징 계산 차이·평가 사람으로 정규화나 종료 epoch 선택은 검산에서 발견하지 않았다.
절대오차 1초 초과의 모든 사례를 errors-over1second.csv에 조건별로 보존했다. 진단 후 결과가 나쁜 사람을 제외하거나 정답을 수정하지 않았다.
기존 속도는 매 시점 이동 속도 절댓값의 평균이다. 순이동거리/3초와 비교한 비율의 중앙값 {diag['speed_definition_audit']['meanabs_dividedby_netspeed_median']:.4f}, 95백분위 {diag['speed_definition_audit']['p95']:.4f}, 최대 {diag['speed_definition_audit']['maximum']:.4f}다. 같은 좌표끼리 비교한 것이므로 독립적인 속도 정답 검증은 아니다.
1명(sub-275)은 영상 투영 방향과 반대인 이동이 90단계 중25단계에 나타났다. 절대속도 평균0.872m/s와 순이동속도0.520m/s가 다르다. 실제 방향 전환인지 좌표 흔들림인지 원본 RGB·보행 단계 주석 없이 확정할 수 없다. 이 구간의 직선 보행 조건은 불확실하므로 다음 입력 품질 점검 대상으로 남긴다. 원래297명 학습·평가에서는 임의로 제외하지 않았다.
TUG≥20초9명의 최대 절대/순이동속도 비율은 약1.013이다. 이 속도 정의 차이만으로 높은 TUG 사례 전체의 큰 오차를 설명하기 어렵다. 또한 이9명 중65세 이상은2명이므로 높은 TUG 결과를 일반 고령자 결과로 동일시하지 않는다.
가장 느린 TUG 사례 {maximum['TUG_seconds']:.2f}초: 별도 보행 속도 {maximum['speed_mps']:.3f}m/s, 속도 기준 예측 {maximum['A']:.2f}초, 기존 특징 {maximum['B']:.2f}초, C {maximum['C']:.2f}초, D {maximum['D']:.2f}초, 결합 E {maximum['E']:.2f}초다.
가능한 설명은 짧은 직선 보행에 의자에서 일어나기·회전·앉기의 어려움이 직접 보이지 않는 점, 20초 이상 사례 9명으로 인한 고득점 범위 압축, 원래 영상 기반 좌표의 측정 오차다. 이번 비교만으로 각 원인의 기여도를 분리하지 못했다.
학습 내부 잔차의 경험적 90% 구간도 diagnosis.json에 저장했다. 이는 정식 보장 구간이나 개인의 임상 변화량이 아니다. 결합 모델은 메타 학습 내부 잔차가 낙관적일 수 있어 이 구간 계산에서 제외했다.

검산 범위
새 내부 저장 모델165개, 최종 저장 모델15개, 결합5개를 모두 재추론했다. 기존 기준 모델15개도 재추론했다.
사람 분리·원본 SHA256·학습 사람만의 정규화/결측 대체/정답 보정·내부 설정 선택·내부 epoch 선택을 확인했다.
기록 예측 재현 최대 차이 {v['max_saved_prediction_difference']:.3g}, 신경망 CPU64 재추론 최대 차이 {v['max_TCN_CPU64_difference']:.3g}초다. CPU64 사전 허용오차2e-5를 바꾸지 않았다.
계산 검산 통과는 실제 휴대폰 추출·일반 고령자 개인 구분·실제 상태 변화 감지의 정확도 입증과 다르다.

보존 위치
작은 결과·설정·표·검산: {H.as_posix()}
원본 참조·새 입력·저장 모델·예측 배열: {B.as_posix()}
기존 모든 연구 파일과 모델은 유지했다. FRAILPOL의 약0.881초는 센서 입력의 별도 개발 결과이며 이번 카메라 유래 자료와 합치지 않았다.
SPPB44명 자료는 별도 보조 연구로 유지하며 이번 표에 새 결과처럼 추가하지 않았다.
추가 입력의 파일 가용성만 점검했다. 같은 방문의5회 일어서기 파일은266명(65세 이상30명), 높은TUG9명 중3명에게 있다. 다른 반복 횟수의 일어서기 파일까지 포함하면273명(65세 이상31명), 높은TUG9명 중6명이다. 없는 파일을 동작 불가능으로 해석하지 않았고 새 학습 입력에도 넣지 않았다. 별도 입력을 추가할 때 누락으로 어려운 사례가 빠지는 편향을 확인해야 한다. 목록 점검 결과는 additional-chair-task-inventory.json에 보존했다.
원자료: https://zenodo.org/records/22309771
'''
    (H/'RESULTS.txt').write_text(record,encoding='utf8')
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.6));arms=['A','B','C','D','D0'] if ablation is not None else ['A','B','C','D','E'];xx=np.arange(len(arms));short=['속도만','특징','관절 CNN','CNN+특징','관절 제거'] if ablation is not None else ['속도만','특징','관절 CNN','CNN+특징','내부 결합']
    for group,offset,color,label in [('all',-.12,'#2563eb','전체 297명'),('age65plus',.12,'#ea580c','65세 이상 34명')]:
        mm=[ablation['summary'][group][a] if a=='D0' else r['summary'][group][a] for a in arms];value=np.array([m['MAE'] for m in mm]);ci=np.array([m['MAE_CI95'] for m in mm]);axes[0].errorbar(xx+offset,value,yerr=[value-ci[:,0],ci[:,1]-value],fmt='o',color=color,capsize=3,label=label)
    axes[0].axhline(1,ls='--',color='#64748b',label='개발 목표 1초');axes[0].set(xticks=xx,xticklabels=short,ylabel='TUG 평균 절대오차 (초)',title='같은 3초 구간·같은 평가 사람');axes[0].set_ylim(bottom=0);axes[0].legend(fontsize=9);axes[0].grid(axis='y',alpha=.2)
    pr=np.load(B/'verified-predictions.npz');old=pr['participant_age']>=65;axes[1].scatter(pr['y'][~old],pr['D'][~old],s=20,alpha=.5,label='65세 미만');axes[1].scatter(pr['y'][old],pr['D'][old],s=28,alpha=.8,label='65세 이상');axes[1].plot([0,45],[0,45],ls='--',color='#64748b');axes[1].set(xlim=(0,45),ylim=(0,45),xlabel='실제 TUG (초)',ylabel='예측 TUG (초)',title='관절 CNN+특징 모델의 개인별 예측');axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    fig.text(.5,.01,'기존 repeat0 5분할의 개발 평가. 오차 막대: 사람 재표집 95% 구간. 실제 휴대폰·MediaPipe 정확도 검증과 구분.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(H/'comparison.png',dpi=160);plt.close(fig)
    inventory=[{'name':path.name,'bytes':path.stat().st_size,'sha256':p.sha(path)} for path in sorted(H.iterdir()) if path.is_file() and path.name not in ['artifact-manifest.json','STATE.json']]
    p.save(H/'artifact-manifest.json',{'small_artifacts':inventory,'large_root':str(B),'input_sha256':p.sha(B/'input.npz'),'verified_prediction_sha256':p.sha(B/'verified-predictions.npz')})
    p.save(H/'STATE.json',{'status':'primary_comparison_verified_sensitivity_required' if r['promising_for_four_repeat_sensitivity'] else 'completed_primary_comparison_and_diagnosis_targets_not_met','goal_passing_arms':r['goal_passing_arms'],'sensitivity_required_arms':r['promising_for_four_repeat_sensitivity'],'clinical_mobile_validation':False,'new_internal_models':165,'new_final_models':15,'blends':5,'original_baselines_replayed':15,'additional_attribution_internal':120 if ablation is not None else 0,'additional_attribution_final':10 if ablation is not None else 0,'report_sha256':p.sha(H/'RESULTS.txt'),'manifest_sha256':p.sha(H/'artifact-manifest.json')})
    print('REPORT_WRITTEN','GOALS',r['goal_passing_arms'],'SENSITIVITY',r['promising_for_four_repeat_sensitivity'],flush=True)

if __name__=='__main__':main()
