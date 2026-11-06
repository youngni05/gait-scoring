"""Write a compact evidence-first research result from verified predictions."""
from pathlib import Path
import json, hashlib
HERE=Path('D:/Projects/D_P/studies/20261103-sppb-tabular-foundation')
BIG=Path('D:/gait-research-data/D_P/sppb-tabular-foundation-20261103')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def delta(v):return f"{v['MAE_difference']:+.3f} [{v['CI95'][0]:+.3f}, {v['CI95'][1]:+.3f}]"
def table(rows):
    return '\n'.join(['| '+' | '.join(map(str,rows[0]))+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows[1:]])
def main():
    results={};verifications=[]
    for scope in ('small','large'):
        v=read(HERE/f'verification-{scope}.json');assert v['status']=='passed' and v['comparison_sha256']==sha(HERE/f'comparison-{scope}.json');verifications.append(v);results.update(read(HERE/f'comparison-{scope}.json'))
    large=results['mobilised'];rows=[['모델','대표 속도 1개','속도 요약 5개','보행 특징 15개','특징−속도 1개, 95% 구간','특징−속도 5개, 95% 구간']]
    for kind in ('LGBM','TabICLv2','TabPFNv2'):
        rows.append([kind,*[f"{large['metrics'][c+'_'+kind]['MAE']:.3f}" for c in ('speed','speed_profile','features')],delta(large['contrasts'][f'features_{kind} minus speed_{kind}']),delta(large['contrasts'][f'features_{kind} minus speed_profile_{kind}'])])
    text=['# SPPB 예측: 기존 자료와 2,232명 확장 비교','',
      '기존 사람 분할을 유지해 TabICLv2·TabPFN v2를 LightGBM과 비교했다. 추가로 Mobilise-D 2,232명의 일상 센서 자료에 같은 비교를 실행했다. 모든 조건의 전원 예측을 다시 재현하고 검산했다. 수치는 SPPB 0–12점에서 평균적으로 얼마나 빗나갔는지 나타내는 평균 절대 오차(MAE)이며 낮을수록 좋다. 차이가 음수이면 앞 조건의 오차가 작다.','',
      f"큰 자료에서 TabICLv2는 속도 요약 5개만 쓸 때 {large['metrics']['speed_profile_TabICLv2']['MAE']:.3f}점, 보행 특징 15개를 쓸 때 {large['metrics']['features_TabICLv2']['MAE']:.3f}점이었다. 추가 특징의 오차 차이는 {delta(large['contrasts']['features_TabICLv2 minus speed_profile_TabICLv2'])}점이다. 따라서 이번 주간 센서 자료에서는 속도를 자세히 요약한 뒤에도 보폭·리듬 정보의 추가 가치가 관찰됐다.",'',
      '반면 기존 37명 자료에서는 두 새 모델에 GDI를 추가해 오차가 줄지 않았다. 큰 자료에는 GDI를 계산할 관절 좌표가 없으므로, 위 결과는 기존 뇌성마비 GDI 모델의 고령자 적용 근거를 직접 확인한 결과가 아니다.','',
      f"평균 오차가 비슷해도 낮은 기능 집단의 판정은 다르다. 실제 SPPB≤8인 920명 중 예측값이 8보다 높아 놓친 사람은 TabICLv2 {large['metrics']['features_TabICLv2']['FN']}명, TabPFN v2 {large['metrics']['features_TabPFNv2']['FN']}명이다. 현재 결과만으로 상태가 나쁜 사람을 충분히 찾아낸다고 말하기 어렵다.",'',
      '## 큰 자료: Mobilise-D 2,232명','',table(rows),'',
      '학습·평가를 사람 단위 5묶음으로 나누었다. 사람마다 센서 자료와 검사 날짜 조건을 충족한 첫 방문 하나만 사용했다. 전체에는 21–95세가 포함되고, 65세 이상은 1,222명이다. 다발성경화증 587명, 만성폐쇄성폐질환 585명, 파킨슨병 573명, 고관절 부위 골절 487명으로 구성된다. 따라서 일반 고령자 표본 또는 짧은 RGB 영상의 검증으로 해석하지 않는다.','',
      '대표 속도는 30초 초과 보행의 주간 속도 90백분위값이다. 속도 요약 5개는 보행 구간 길이별 속도 평균·상위 속도·변동성이고, 전체 15개는 여기에 보폭·걸음 빈도·한 주기 시간의 평균과 변동성을 더한다. 임상 검사 때 잰 4 m 보행 속도, 연령, 질환명, SPPB 하위 점수는 모델 입력에서 제외했다. 주간 요약값으로부터 짧은 보행의 좌우 협응을 새로 계산한 실험은 아니다.','']
    rows=[['65세 이상 1,222명','속도 1개 MAE','속도 5개 MAE','특징 15개 MAE','특징−속도 5개, 95% 구간']]
    for kind in ('LGBM','TabICLv2','TabPFNv2'):
        rows.append([kind,*[f"{large['subgroups']['age65plus'][c+'_'+kind]['MAE']:.3f}" for c in ('speed','speed_profile','features')],delta(large['subgroup_contrasts']['age65plus'][f'features_{kind} minus speed_profile_{kind}'])])
    text.extend([table(rows),'','65세 이상 표는 전체 연령으로 학습한 모델의 해당 평가 사람들만 집계한 결과다. 65세 이상만 별도로 학습한 결과는 아니다.','', '## 같은 입력에서 모델 변경의 효과',''])
    rows=[['비교: 특징 15개','MAE 차이와 사람 단위 95% 구간']]
    for a,b in [('TabICLv2','LGBM'),('TabPFNv2','LGBM'),('TabICLv2','TabPFNv2')]:rows.append([f'{a} − {b}',delta(large['contrasts'][f'features_{a} minus features_{b}'])])
    text.extend([table(rows),'','## 기존 소규모 자료',''])
    for name,title in [('older37','3D 보행·GDI 동시 사용 가능 37명'),('older24','충분한 보행 주기 조건을 충족한 24명'),('older44','3D 보행 전체 44명'),('gait2care','GAIT2CARE 92명'),('gstride','GSTRIDE 158명')]:
        r=results[name];rows=[['모델','속도만 MAE','보행 특징 MAE','GDI 추가 MAE','특징−속도, 95% 구간']]
        for kind in ('LGBM','TabICLv2','TabPFNv2'):
            rows.append([kind,f"{r['metrics']['speed_'+kind]['MAE']:.3f}",f"{r['metrics']['features_'+kind]['MAE']:.3f}",f"{r['metrics']['features_GDI_'+kind]['MAE']:.3f}" if 'features_GDI_'+kind in r['metrics'] else '해당 없음',delta(r['contrasts'][f'features_{kind} minus speed_{kind}'])])
        text.extend(['### '+title,'',table(rows),''])
        if name in ('older37','older24'):
            rows=[['GDI 추가 효과','GDI 추가−보행 특징, 95% 구간']]
            for kind in ('LGBM','TabICLv2','TabPFNv2'):rows.append([kind,delta(r['contrasts'][f'features_GDI_{kind} minus features_{kind}'])])
            text.extend([table(rows),''])
    text.extend(['37명·24명은 44명의 부분집합이므로 독립된 세 연구로 합산하지 않는다. GAIT2CARE·GSTRIDE·Mobilise-D는 센서 자료이며, 기존 GDI 모델이 요구하는 전신 관절 좌표를 제공하지 않아 GDI 추가 조건을 만들지 않았다. 37명·24명의 GDI는 이전에 고정한 가상 측면 카메라 투영과 모델의 예측값이다. 발끝·관절 중심 정의 차이 및 실제 RGB 영상의 좌표 오차는 해결된 것으로 취급하지 않는다.','','## 낮은 SPPB와 큰 오차',''])
    rows=[['Mobilise-D 조건','전체 MAE','0–4점 MAE','5–8점 MAE','9–10점 MAE','11–12점 MAE','2점 초과 오차','SPPB≤8 놓침 / 실제≤8','오경보 / 실제>8']]
    for kind in ('LGBM','TabICLv2','TabPFNv2'):
        for c in ('speed','speed_profile','features'):
            m=large['metrics'][c+'_'+kind];rows.append([c+'_'+kind,f"{m['MAE']:.3f}",*[f"{m['bands'][b]['MAE']:.3f}" for b in ['0-4','5-8','9-10','11-12']],m['errors_gt2'],f"{m['FN']} / {m['TP']+m['FN']}",f"{m['FP']} / {m['TN']+m['FP']}"])
    text.extend([table(rows),'','낮은 점수 판정은 예측 SPPB≤8로 고정한 참고 집계다. 이번 자료에서 유리한 기준점을 고르지 않았으며 임상 선별검사로 검증한 것은 아니다.','','## 질환별 결과',''])
    rows=[['질환','사람 수','LGBM 속도→특징','TabICLv2 속도→특징','TabPFNv2 속도→특징']]
    for disease in ('COPD','MS','PD','PFF'):
        g=large['subgroups'][disease];rows.append([disease,g['Mean']['n'],*[f"{g['speed_'+k]['MAE']:.3f} → {g['features_'+k]['MAE']:.3f}" for k in ('LGBM','TabICLv2','TabPFNv2')]])
    text.extend([table(rows),''])
    rows=[['질환','TabICLv2 특징−속도 5개, 95% 구간','TabPFN v2 특징−속도 5개, 95% 구간']]
    for disease in ('COPD','MS','PD','PFF'):rows.append([disease,*[delta(large['subgroup_contrasts'][disease][f'features_{k} minus speed_profile_{k}']) for k in ('TabICLv2','TabPFNv2')]])
    text.extend([table(rows),'','TabICLv2의 속도 외 특징 추가 효과는 COPD·MS 집단에서 관찰됐다. PD·PFF 집단에서는 해당 차이의 95% 구간에 0이 포함돼 뚜렷한 개선을 확인하지 못했다. 전체 평균의 개선이 모든 질환에 고르게 적용되는 것은 아니다.','', '질환별 분석은 같은 모델의 평가 결과를 나눈 설명용 분석이다. 질환을 학습에서 통째로 제외하는 외부 검증은 수행하지 않았다. 보행 특징이 질환의 차이를 간접적으로 표현했을 가능성도 배제하지 못하므로 일반 고령자 내 기능 차이를 확인한 것으로 확대해석하지 않는다.','','## 검산과 해석 범위','',
      f"- 새 최종 모델 {sum(v['reconstructed_models'] for v in verifications)}개의 모든 평가 사람을 재예측했다. 최대 재현 차이는 {max(v['maximum_replay_error'] for v in verifications):.8g}점이다. 기초 모델은 저장한 학습 문맥과 동일한 고정 가중치로 재구성했고, LightGBM은 저장 모델을 다시 읽고 나무 구조도 독립 계산했다.",
      '- 소규모 비교는 기존 특징·사람·분할·LightGBM 결과의 파일 지문을 대조했다. 확장 LightGBM의 설정은 각 평가 사람을 제외한 내부 3묶음에서만 골랐다. 두 기초 모델은 각 8개 앙상블과 고정 초기값을 사용했다. 새 신경망 가중치 학습이나 평가 정답을 이용한 보정은 없다.',
      '- LightGBM은 기존 비교와 같은 최대 깊이 3, 나무 50/150개 등 16개 설정을 탐색했다. 두 새 모델은 고정 설정을 썼다. 따라서 이번 조건의 비교이며 모든 가능한 LightGBM 설정보다 우월하다는 결론은 아니다. 모델 가중치 버전은 TabICLv2와 TabPFN v2이고, 실행 라이브러리 버전과 구분해 기록했다.',
      '- 95% 구간은 같은 사람끼리 오차 차이를 비교해 10,000회 재표집했다. 이미 얻은 평가 예측을 조건으로 한 구간이며, 학습 전체를 반복할 때의 변동까지 포함하지 않는다. 다수의 탐색 비교와 기존 개발 자료 재사용이 있으므로 독립 외부 확인이 필요하다.',
      '- Mobilise-D 2,230명은 하위 점수 합과 총점이 일치했다. 나머지 2명은 공식 총점 8점이 있으나 하위 점수 하나가 누락돼 합 검산을 할 수 없다. 임의의 하위 점수를 만들지 않았다. 대표 속도 누락 34명과 이 2명을 각각 제외한 평가 집계도 comparison-large.json에 저장했다.',
      '- 같은 SPPB 정답을 쓰더라도 주간 센서 자료의 개선은 짧은 카메라 영상의 개선을 보장하지 않는다. SPPB 자체에 보행 항목이 포함되므로 점수와 관련된다는 사실만으로 전반적인 신체 상태나 미래 낙상을 예측한다고 말할 수 없다.',
      '- 큰 자료와 작은 자료는 대상자·센서·측정 시간·특징이 함께 다르다. 따라서 두 자료의 결과 차이를 사람 수가 늘어난 효과만으로 설명할 수 없다.',
      '- 원자료 Mobilise-D의 비상업·변경금지 이용 조건은 제품 사용 허가를 뜻하지 않는다. 본 결과는 내부 연구 비교다.',
      '', '## 더 큰 일반 고령자 자료의 다음 후보','',
      '[FRAILPOL](https://www.nature.com/articles/s41597-026-06854-8)은 지역사회 거주 668명의 TUG 중 5개 착용 센서 기록과 노쇠 분류를 공개한다. 일반 고령자의 짧은 동작이라는 점은 더 가깝지만, 이번 SPPB 회귀 및 RGB/GDI 실험과 정답·측정 방식이 다르다. 이번 실행에는 추가하지 않았다.',
      '', '[688명 RGB 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306775/)도 확인했으나 원영상과 관절 좌표는 공개되지 않고 저자 요청이 필요하다. 또한 정면에서 SPPB 동작 전체를 찍고 자체 기능 위험 지수를 예측한다. 기존 측면 GDI 입력이나 순수한 짧은 보행 SPPB 회귀에 곧바로 사용할 수 있는 자료로 세지 않았다.',
      '', '## 재현 위치','',f'- 조건·결과·검산: `{HERE}`',f'- 고정 가중치·학습 문맥·예측·나무 모델: `{BIG}`','- `protocol.json`, `weights-manifest.json`, `requirements-lock.txt`, `verification-small.json`, `verification-large.json`에 조건과 파일 지문이 있다.'])
    (HERE/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    manifest={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='status.json'}
    (HERE/'status.json').write_text(json.dumps(dict(status='completed_and_verified',models=sum(v['reconstructed_models'] for v in verifications),artifacts=manifest),indent=2),encoding='utf8')
    print('RESULTS written and verified input hashes matched')
if __name__=='__main__':main()
