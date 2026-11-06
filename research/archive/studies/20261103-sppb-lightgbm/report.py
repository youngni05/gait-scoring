from common import *
def main():
    r=read(HERE/'comparison.json');v=read(HERE/'verification.json');inv=read(HERE/'inventory.json')
    assert v['comparison_sha256']==sha(HERE/'comparison.json')
    labels={'older37':'관절 자료 공통 37명','older24':'충분한 반복 주기 24명','older44':'관절 자료 전체 44명','gait2care':'GAIT2CARE 92명','gstride':'GSTRIDE 158명'}
    text='''# 보행 특징과 GDI를 추가한 LightGBM의 SPPB 예측 비교

보행 특징을 늘려 LightGBM을 학습하고, 속도만 사용하는 경우 및 기존 예측 GDI를 추가하는 경우를 같은 사람 분할로 비교했다. **37명 관절 자료에서는 속도 단독보다 오차가 커졌고 GDI의 추가 이득도 확인되지 않았다. GSTRIDE 158명에서는 평균오차가 조금 줄었지만 불확실성 구간이 개선 없음도 포함했다.**

이번 실험은 공개 SPPB 총점에 대한 사람 분리 개발 평가다. 새 사람에게 예측을 내도록 분리했지만, 반복해서 사용한 자료이므로 독립 외부검증이나 일반 고령자 대상 임상 검증은 아니다.

**현재 보유 자료 규모**

| 자료 | 보행과 SPPB가 연결된 사람 | 이번 사용 | 기존 GDI 추가 |
|---|---:|---|---|
| OlderAdults44 | 44명, 220회 기록 | 44명 전체, 공통 37명, 반복 주기 충분한 24명 | 37명에서 가상 촬영 입력으로 가능 |
| GAIT2CARE | 92명, 등록 127명 | 운동 전 Week0만 | 전신 관절·영상이 없어 불가 |
| GSTRIDE | 158명, 등록 163명 | 한 번의 방문 | 전신 관절·영상이 없어 불가 |
| Mobilise-D | 2,232명 | 규모만 재확인, 이번 재학습 제외 | 주간 센서 요약이며 불가 |

37명과24명은44명의 부분집합이다.44명 중65세 이상은31명,37명 중28명,24명 중18명이다. GAIT2CARE와 GSTRIDE 사이에는 동일인 여부가 미해결인 후보가 있어 모든 자료를 독립 인원으로 단순 합산하거나 합쳐 학습하지 않았다. Mobilise-D는 여러 날의 센서 요약과 여러 질환군이 포함된 자료이며 짧은 보행 영상 자료와 구분한다.

**무엇을 입력하고 학습했는가**

모델은 별도로 측정된 실제 SPPB 총점을 예측했다. SPPB 검사 자체의4m 속도, 균형·의자 일어나기 점수, TUG, 나이·질환명은 입력하지 않았다. 논문의123개 영상 특징을 그대로 복제한 것이 아니라 보유 자료에서 관찰할 수 있는 보행 특징을 확장했다.

| 자료 | 확장 입력 개수 | 주요 내용 |
|---|---:|---|
| 관절37명·44명 | 25개 | 속도, 보행 주기·보폭·보행 폭·유각 비율·걸음 빈도, 무릎·다리·발의 움직임 범위, 발끝·골반의 상하 움직임, 발목의 상대 이동 속도; 양측 평균과 좌우 차이 |
| 관절24명 | 29개 | 위25개에 기존 PCI와 주기·보폭·유각 비율 변동계수4개 추가 |
| GAIT2CARE | 53개 | 속도·기존 변동성·PCI, 발 들림·경로·기울기·각속도·지지 단계 등16개 원본 지표의 평균·발별 분산을 합친 표준편차·좌우 차이 |
| GSTRIDE | 36개 | 속도·기존 변동성 및 위16개 지표의 평균·표준편차; 한쪽 발 자료이므로 좌우 차이·PCI 제외 |

관절 움직임 범위는 주기 내95백분위수−5백분위수이며 임상 관절각 모델의 해부학적 ROM과 동일하다고 보지 않는다. 발목 속도는 골반에 대한 상대 전방 속도이며100Hz에서11프레임,2차 평활 미분을 사용했다. 발끝 상하 범위도 실제 지면 최소 여유 높이와 구분한다. 팔·몸통·회전 특징은 관절 목록과 직선 과제에서 확보되지 않아 추가하지 않았다.

5회 기록의 경계를 연결하지 않고 실제 공개 착지 사건 사이에서 특징을 계산했다.37명 주분석은 반복수가 민감한 PCI/CV를 제외하고 평균·주기 내 움직임을 사용했다.24명 분석에서만 양쪽 각각15주기 이상이라는 기존 조건에 따라 PCI/CV를 포함했다.15주기는 본 탐색의 조건이며 모든 지표의 측정 신뢰도를 보증하지 않는다.

센서 자료는 기존 직선 걸음 필터를 그대로 적용했다. 추가 특징별 유효값이 충분한지 확인했으며 GAIT2CARE1명의 PCI만 결측이었다. LightGBM의 결측 처리를 사용했다. 양발 지지율은 이전 재현이 해결되지 않아 제외했다. GSTRIDE1명은 원본 하위 검사에 'Incapable'이 있어 합계 검산 불가로 기록하고, 배포된 총점을 그대로 사용했다. 이를 임의로0점으로 바꾸지 않았다.

**같은 사람 기준 예측 성능**

단위는 SPPB 점수의 평균 절대오차(MAE)이며 작을수록 좋다. 확장 입력에는 속도가 포함된다. GDI 추가는 확장 입력에 기존 고정 모델의 예측값1개를 더했다.

| 자료 | 속도 LightGBM | 확장 특징 LightGBM | 확장 특징과 GDI LightGBM | 속도 Ridge 대조 |
|---|---:|---:|---:|---:|
'''
    for n,out in r.items():
        m=out['metrics'];a=[f"{m[k]['MAE']:.3f}" if k in m else '계산 불가' for k in ['speed_LGBM','features_LGBM','features_GDI_LGBM','speed_Ridge']]
        text+='|'+labels[n]+'|'+'|'.join(a)+'|\n'
    text+='''
Ridge는 기존 단순 회귀 기준을 함께 유지하기 위한 대조다. LightGBM의 속도 단독이 약해져 상대적으로 좋아 보이는 경우를 구분한다.37명과24명에서는 기존 속도 Ridge 결과도 재현됐다.

**추가 특징과 GDI의 효과 크기**

오차 차이는 앞 모델−기준 모델이다. 음수면 앞 모델이 좋다.95% 구간은 고정된 교차검증 예측을 사람 단위로10,000번 재표집한 조건부 구간이며, 전체 학습을 반복한 불확실성이나 다중 비교 보정 구간은 아니다.

| 자료 | 비교 | MAE 차이 | 95% 구간 |
|---|---|---:|---|
'''
    for n,out in r.items():
        for c,title in [('features_LGBM-speed_LGBM','확장 특징−속도'),('features_GDI_LGBM-features_LGBM','GDI 추가−확장 특징'),('features_LGBM-speed_Ridge','확장 특징−속도 Ridge')]:
            if c in out['contrasts']:
                d=out['contrasts'][c];lo,hi=d['CI95'];text+=f"|{labels[n]}|{title}|{d['MAE_difference']:+.3f}|{lo:+.3f}~{hi:+.3f}|\n"
    text+='''
GSTRIDE의 확장 특징은 속도 LightGBM보다0.052점 낮은 오차였지만 구간에0이 포함된다.37명의 GDI 추가 차이는−0.005점에 그쳤다. 이번 결과는 보행 특징이나 GDI가 언제나 무의미하다는 결론이 아니라, 이번 자료·추출법·모델 설정에서 추가 이득이 확실하지 않았다는 뜻이다.

**낮은 기능과 큰 오차**

SPPB≤8 및 예측≤8이라는 고정 기준의 탐색적 집계다. 보편적인 임상 진단 성능으로 검증된 기준이 아니다.

| 자료 | 모델 | 실제 낮은 기능 중 발견 | 놓침 | 그 외 사람 중 오경보 | 2점 초과 오차 |
|---|---|---|---:|---|---:|
'''
    for n,out in r.items():
        for k,title in [('speed_LGBM','속도'),('features_LGBM','확장 특징'),('features_GDI_LGBM','확장 특징과 GDI'),('speed_Ridge','속도 Ridge')]:
            if k not in out['metrics']:continue
            m=out['metrics'][k];text+=f"|{labels[n]}|{title}|{m['TP']}/{m['TP']+m['FN']}|{m['FN']}|{m['FP']}/{m['FP']+m['TN']}|{m['errors_gt2']}|\n"
    text+='\n| 자료 | 모델 | 0–4점 MAE와 편향 | 5–8점 MAE와 편향 | 9–10점 MAE와 편향 | 11–12점 MAE와 편향 |\n|---|---|---|---|---|---|\n'
    for n,out in r.items():
        for k,title in [('speed_LGBM','속도'),('features_LGBM','확장 특징'),('features_GDI_LGBM','확장 특징과 GDI')]:
            if k not in out['metrics']:continue
            cells=[]
            for b in out['metrics'][k]['bands'].values():cells.append(f"{b['MAE']:.3f} / {b['bias']:+.3f} (n={b['n']})" if b['n'] else '자료 없음')
            text+='|'+labels[n]+'|'+title+'|'+'|'.join(cells)+'|\n'
    text+='''
편향이 양수면 실제보다 기능을 높게 예측했다. GSTRIDE의0–4점12명에서는 확장 모델도 평균2.780점 높게 예측했다. GAIT2CARE의 같은 점수대15명은3.367점 높게 예측했다. 평균오차가 줄더라도 낮은 기능의 사람을 놓치는 문제가 해결된 것은 아니다.

**촬영 구도와 GDI 해석**

기존37명 결과의 가상 카메라 투영을 그대로 사용했다. 고정 지점에서 사람을 따라 좌우 회전하는 측면 구도, 거리3.5m, 높이1m, 수평 화각60도,29.97Hz, 연속124프레임 조건이다. 높이·화각은 가정이다. 원 전처리의 좌표 방향과 크기 보정을 유지했으며 짧은 기록을 늘리거나 채우지 않았다.

54개 CP 모델의 학습·보정값은 고정했다. 저장된 모델별 출력을 다시 결합해37명의 예측 GDI가 같은지 확인했다. 이번에는 CP 모델을 새로 학습하거나54개 모델의 순전파를 새로 실행하지 않았다. 발끝 대용 마커와 BODY25 지점의 동일성, 가상 좌표와 실제 RGB 자세 추정의 차이는 남아 있다. 따라서 GDI 추가 조건은 모사 입력의 탐색이며 고령자의 실제 GDI 정확도 또는 카메라 서비스 검증이 아니다.

**검증 방법과 완료 범위**

사람마다 한 행을 만들고5개 묶음을 번갈아 평가했다.44명 자료는 기존 분할을 부분집합에서도 유지했다. 큰 두 자료는 기존의 사람ID 정렬과 난수로5분할을 재현했다. 각 평가 묶음의 정답을 가린 상태에서 나머지 사람을 다시3개로 나눠 설정을 골랐다.

LightGBM은 깊이3, 학습률0.03을 고정하고 잎3/7개·잎 최소3/8명·규제1/10·나무50/150개의16가지 조합을 내부 MAE로 선택했다. 조건마다 같은 탐색 범위를 사용했고 평가 사람을 이용한 조기 종료·특징 선택·점수 보정은 하지 않았다. 예측을0~12로 강제 절단하지 않았다. [LightGBM 공식 설정 문서](https://lightgbm.readthedocs.io/en/stable/Parameters.html)

3,340개 학습 기록의 사람 분리와 내부 설정 선택을 검산했다. 최종 모델85개를 불러와 재추론했고, 나무의 분기값을 직접 따라 계산하거나 Ridge 계수를 독립적으로 풀어 예측을 대조했다. 최대 차이는1.60×10⁻¹⁴점이었다. 내부 탐색의 모든 나무를 새로 재학습한 것은 아니다.

원자료·중간자료478개 파일의 SHA256과 원본 SPPB 표를 확인했다. 관절 특징의 사람별 집계1,056개, 별도 각도 공식에 의한40개 주기, CSV를 별도로 읽은 센서 통계240개를 검산했다. GDI 체크포인트54개의 지문과 저장 출력의 결합도 확인했다. 이는 계산 재현이며 센서나 마커 좌표의 물리적 정확도 검증은 아니다.

65세 이상만 골라 본 관절37명 평가의28명 하위집단에서는 속도 LightGBM1.165점, 확장 특징1.364점, GDI 추가1.367점이었다. 고령자만 따로 재학습한 모델은 아니다. 관절 자료에는SPPB0–4점이 없고, 센서 자료의 긴 보행을 짧은 영상과 동일시할 수 없다.

계산 기준은protocol.json, 사람별 예측·모든 점수대는comparison.json, 검산 결과는verification.json에 보존했다. 코드의 핵심 흐름은prepare.py가 사람별 보행 특징을 만들고fit.py가 학습 사람 안에서 모델을 선택하며verify.py가 저장 모델과 점수를 다시 확인하는 구조다.

작은 산출물은D:/Projects/D_P/studies/20261103-sppb-lightgbm, 모델·특징·학습 기록은D:/gait-research-data/D_P/sppb-lightgbm-20261103에 저장했다. 기존 데이터·결과·GPU 작업·예약은 변경하지 않았다.
'''
    (HERE/'RESULTS.md').write_text(text,encoding='utf8')
    status=dict(owner_thread='01a19c6f-49f7-7300-97b0-a303b71ffe91',status='COMPLETED_WITH_LIMITS',completed=['Expanded features from220 optical trials and250 IMU participants','Five matched-person nested comparisons','Frozen GDI incremental comparison in37 and24 people','85 final models independently replayed','Existing Mobilise-D2232-person inventory verified'],not_performed=['GDI inference for IMU-only datasets: inputs unavailable','Mobilise-D LightGBM retraining: weekly-summary inventory only','New CP/GDI training or new GPU jobs','Clinical or RGB measurement validation'],limitations=['Repeated-use development cohorts; one fixed outer split','Conditional fixed-prediction uncertainty','Older44 includes54-64year participants; noSPPB0-4','GDI based on virtual-camera inputs and toe proxy','Cross-dataset participant independence unresolved','One GSTRIDE published total retained without subscore sum verification'],artifacts={f:sha(HERE/f) for f in ['protocol.json','inventory.json','source-manifest.json','preparation-receipt.json','fit-receipt.json','comparison.json','verification.json','RESULTS.md']},large_data_path=str(BIG))
    save(HERE/'status.json',status)
    save(HERE/'progress.json',dict(status='completed_and_verified',fits=3340,final_models=85,verification_sha256=sha(HERE/'verification.json')))
    print('Report saved; verification digest matched.',flush=True)
if __name__=='__main__':main()
