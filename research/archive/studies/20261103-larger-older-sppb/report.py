"""Human-readable result and a scientific comparison figure, from verified saved outputs."""
from prepare import *
from collections import Counter
LABELS={'A':'속도만','B':'속도 + 좌우 협응','C':'속도 + 변동성','D':'속도 + 협응 + 변동성','Mean':'학습 사람 평균','Median':'학습 사람 중앙값'}
def main():
 r=read(S/'comparison.json');p=read(S/'people.json');a=read(S/'input-audit.json');v=read(S/'verification.json');lines=['# 더 큰 고령자 자료의 SPPB 예측 비교','', '실험 질문: 별도의 보행 기록에서 얻은 속도에 좌우 협응·걸음 변동성을 더하면, 처음 보는 고령자의 실제 SPPB 점수를 더 잘 예측하는가?','', '입력은 발 부착 센서로 계산한 보행 지표이며 정답은 공개된 실제 SPPB 점수다. 기존 CP/GDI 모델은 사용하지 않았다. 이 자료에는 해당 모델에 필요한 전신 관절 좌표나 측면 영상이 없다.','', '## 결과','', '이번 고정 조건에서는 협응·변동성을 추가했을 때 SPPB 예측 오차가 줄어든다는 근거가 확인되지 않았다. 속도 기준 자체는 학습 집단 평균값보다 오차가 작았지만, 낮은 기능 점수를 실제보다 높게 예측하는 문제가 남았다.','', '| 자료·사람 수 | 속도만 | +협응 | +변동성 | +협응·변동성 | 학습 평균 기준 |','|---|---:|---:|---:|---:|---:|']
 for ds in ['gait2care','gstride']:
  x=r[ds+'/sppb'];m=x['metrics'];val=lambda c:f"{m[c]['MAE']:.3f}" if c in m else '계산 불가'
  lines.append(f"| {ds.upper()} · {len(x['ids'])}명 | {val('A')} | {val('B')} | {val('C')} | {val('D')} | {val('Mean')} |")
 lines+=['','단위는 SPPB 점수의 평균 절대오차다. 낮을수록 좋다.','', '| 비교 | 추가 지표 모델 − 속도 기준 오차 | 사람 단위 95% 구간 |','|---|---:|---|']
 for ds in ['gait2care','gstride']:
  for name,d in r[ds+'/sppb']['contrasts'].items():lines.append(f"| {ds} {name} | {d['MAE_difference']:+.3f} | [{d['CI95'][0]:+.3f}, {d['CI95'][1]:+.3f}] |")
 lines+=['','음수는 추가 지표가 유리하다는 뜻이다. 구간이 0을 포함하면 우연한 분할·표본 차이로 설명될 가능성을 배제하기 어렵다. 이 구간은 저장된 평가 예측에 대해 사람을 재추출한 조건부 구간이며, 전체 모델 재학습의 불확실성은 포함하지 않는다.','', '## 낮은 점수와 큰 오차','', '| 자료 | 조건 | SPPB≤8 발견/전체 | 높은 점수를 낮다고 판단 | 2점 초과 오차 |','|---|---|---|---|']
 for ds in ['gait2care','gstride']:
  for c in ['A','D' if ds=='gait2care' else 'C']:
   m=r[ds+'/sppb']['metrics'][c];lines.append(f"| {ds} | {LABELS[c]} | {m['TP']}/{m['TP']+m['FN']} | {m['FP']}/{m['FP']+m['TN']} | {m['errors_gt2']}/{m['n']} |")
 lines+=['','예측값≤8이라는 고정 기준으로 집계한 탐색 결과이며 진단 성능으로 검증된 기준은 아니다.','', '| 자료 | 조건 | 0–4점 MAE (인원) | 5–8점 | 9–10점 | 11–12점 |','|---|---|---|---|---|']
 for ds in ['gait2care','gstride']:
  for c in ['A','D' if ds=='gait2care' else 'C']:
   bands=r[ds+'/sppb']['metrics'][c]['bands'];cells=[f"{bands[k]['MAE']:.3f} ({bands[k]['n']})" if bands[k]['n'] else '없음' for k in ['0-4','5-8','9-10','11-12']];lines.append('| '+ ' | '.join([ds,LABELS[c],*cells])+' |')
 lines+=['','## 대상과 계산 방법','', f"- 등록 인원은 GAIT2CARE {a['registered']['gait2care']}명, GSTRIDE {a['registered']['gstride']}명이다. 보행 파일과 유효한 점수가 연결된 인원은 각각 {a['included']['gait2care']}명, {a['included']['gstride']}명이다.",'- GAIT2CARE의 운동 전 Week0만 사용했다. 운동 후 자료를 별도 사람처럼 늘리지 않았다. 총점과 세부 점수 합이 맞지 않는 1명은 정답을 수정하지 않고 제외했다.','- GAIT2CARE 92명 중 1명은 한쪽 발의 유효한 좌우 위상 쌍이 5개로 최소 15개에 못 미쳐 네 조건 공통 비교는 91명이다. 원본의 좌우 걸음 수가 다른 경우 끝의 0 채움 행은 실제 사건으로 계산하지 않았다.','- 모든 분석 대상은 65세 이상이다. GSTRIDE 나이는 구간으로 공개돼 있어 정확한 나이로 해석하지 않았다.','- 속도는 긴 보행 기록의 걸음별 속도 평균이다. 정답 SPPB에 직접 포함되는 4m 검사 속도는 입력에 넣지 않았다.','- 회전 각도 절댓값 20도 미만이며 시간·길이·속도가 양수이고 흔들기 비율이 0~100 사이인 걸음을 사용했다. 표본을 임의로 지우는 추가 이상치 보정은 하지 않았다.','- 변동성은 같은 발 안의 보행 주기·보폭·발을 흔드는 시간 비율의 변동계수다. 좌우 평균의 차이가 변동성으로 섞이지 않게 계산했다.','- 협응은 보정 전 공통 시간값에 연결한 좌우 발 디딤의 위상 협응 지수(PCI)다. 제작자 설명에 따라 센서별 보정 시간은 좌우 정렬에 사용하지 않았다. 사건 번호가 1부터 시작함을 원본 보행 주기 시간과 대조했다.','- 다섯 사람 묶음을 번갈아 평가했다. 각 묶음을 평가할 때 나머지 사람 안에서만 세 묶음으로 설정을 선택했다. 모든 조건은 동일한 사람과 분할을 사용했다.','- 같은 단순 회귀 모형(Ridge)을 사용하고 강도 0.1,1,10,100,1000 중 학습 사람 안에서만 골랐다. 예측 점수를 0~12로 강제로 자르지 않았다.','', '## 지역사회 거주자만의 평가','', '| 자료 | 인원 | 속도만 MAE | 추가 지표 MAE |','|---|---:|---:|---:|']
 for ds in ['gait2care','gstride']:
  m=r[ds+'/sppb']['community_subgroup'];c='D' if ds=='gait2care' else 'C';lines.append(f"| {ds} | {m['A']['n']} | {m['A']['MAE']:.3f} | {m[c]['MAE']:.3f} |")
 extra=r['gait2care_available_AC/sppb'];ed=extra['contrasts']['C-A'];lines += ['',f"협응 계산 불가 1명을 포함한 92명의 속도/변동성 비교에서는 MAE {extra['metrics']['A']['MAE']:.3f} → {extra['metrics']['C']['MAE']:.3f}, 차이 {ed['MAE_difference']:+.3f}, 95% 구간 [{ed['CI95'][0]:+.3f}, {ed['CI95'][1]:+.3f}]였다. 이 비교도 개선이 확인된 것은 아니다."]
 lines+=['','위 표는 전체 집단으로 시행한 사람 분리 평가 예측 중 지역사회 거주자를 골라 본 결과다. 지역사회 거주자만으로 따로 학습한 실험은 아니다.','', '## 다른 자료 적용 — 독립성 미확인','']
 x=r['provisional_transfer/sppb'];d=x['contrasts']['C-A'];lines+=[f"GSTRIDE {len(x['source_ids'])}명에서만 학습·설정을 선택한 뒤 GAIT2CARE {len(x['ids'])}명에 적용했다. 속도만 MAE {x['metrics']['A']['MAE']:.3f}, 변동성 추가 MAE {x['metrics']['C']['MAE']:.3f}, 차이 {d['MAE_difference']:+.3f}점, 95% 구간 [{d['CI95'][0]:+.3f}, {d['CI95'][1]:+.3f}]이다.",'','두 자료의 가명 식별자는 서로 연결되지 않는다. 성별·키·체중이 같은 후보 조합은 '+str(len(a['exact_demographic_fingerprint_pairs']))+'건이며, 동일인인지 확인되지 않았다. 공개 파일의 중복 여부 확인만으로 같은 사람이 없다고 보장할 수 없다. 따라서 위 결과를 독립 외부검증으로 부르지 않는다. 한쪽 발과 양발, 보행 장소·대상 구성도 서로 다르다.','', '## 탐색 분석: 어떤 기능과 관련되는가','', '| 자료 | 정답 | 속도만 MAE | 협응·변동성 추가 MAE |','|---|---|---:|---:|']
 for ds in ['gait2care','gstride']:
  for target,label in [('balance','균형 점수'),('chair','의자 일어나기 점수'),('tug','TUG 초')]:
   m=r[ds+'/'+target]['metrics'];c='D' if ds=='gait2care' else 'C';lines.append(f"| {ds} | {label} | {m['A']['MAE']:.3f} | {m[c]['MAE']:.3f} |")
 lines+=['','탐색 분석이며 여러 정답을 확인한 뒤 유리한 결과만 주결론으로 선택하지 않는다. 균형·의자 점수는 각각 0~4점, TUG는 초 단위다.','', '## 검증과 한계','',f"- 저장된 {v['saved_fits']}개 학습을 재계산하고, 최종 {v['final_models']}개 모형의 저장 예측을 재현했다. 독립 계산과 최대 예측 차이 {v['max_closed_form_prediction_error']:.3g}.",f"- 원본 {v['source_files_rehashed']}개 파일의 SHA256을 재확인했다. 점수표는 배포 MD5, 압축파일에서 가져온 개별 파일은 ZIP CRC32를 대조했다. 압축 전체를 다운로드하지 않았으므로 전체 ZIP MD5를 검증했다고 주장하지 않는다.",'- 보조 시간 대조: 원본 공통 시간의 발 접촉 겹침으로 계산한 양발 지지율과 제작자 제공값을 3명에서 비교했을 때 절대 차이 중앙값은 0.36~0.76%p, 최대 12.18%p였다. 동일 산식·처리 조건인지 확인되지 않아 양발 지지율의 정확한 재현 또는 PCI 측정 정확도를 입증한 것으로 보지 않는다.','- 원본 파일 보존·계산 재현은 센서 측정 정확도 검증과 다르다. 같은 신호에서 제작자가 추정한 발 디딤과 보행 지표를 사용했으며 독립적인 광학 정답과 비교하지 않았다.','- 오래 걷는 동안의 속도 변화·멈춤·환경 변화가 변동성에 섞일 수 있다. 짧은 측면 영상의 수치와 동일하게 볼 수 없다.','- CP GDI 모델의 고령자 정확도, 단일 카메라 측정 정확도, 한국 일반 고령자 전체에 대한 성능을 입증한 결과가 아니다.','', '## 원자료와 저장 위치','', '- [GAIT2CARE v2](https://zenodo.org/records/17017009), [GSTRIDE v3](https://zenodo.org/records/17052815). CC BY 4.0. 각 배포 페이지의 저자·출처를 유지했다.',f'- 원본·큰 중간자료·저장 모형: `{B}`.', '- 실행 기준: protocol.json. 사람 연결·제외: input-audit.json. 상세 결과: comparison.json. 재계산: verification.json.']
 (S/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('REPORT_WRITTEN',S/'RESULTS.md')
if __name__=='__main__':main()
