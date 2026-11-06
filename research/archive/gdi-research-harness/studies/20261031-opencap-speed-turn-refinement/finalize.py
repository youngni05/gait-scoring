from study import *
import platform,py_compile
from collections import Counter

def main():
    v=json.loads((HERE/'results-verification.json').read_text());assert v['passed'] and v['models']==125 and v['inner_fits']==1500
    assert json.loads((HERE/'input-verification.json').read_text())['passed']
    summary=json.loads((HERE/'summary.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text())
    names={'Current':'기존 단순 Ridge','Inverse':'평균 보행 속도의 역수 + Ridge','SameTUG':'같은 TUG 보행 속도 + Ridge'}
    def metric(c,view,band):return next(r for r in summary[c]['metrics'] if r['view']==view and r['band']==band)
    def table(c):
        lines=['| 모델 | 전체 절대오차 | 긴 8명 절대오차 | 긴 8명 과소예측 | 긴 8명 과대예측 |','|---|---:|---:|---:|---:|']
        for view in protocol['cohorts'][c]['views']:
            a=metric(c,view,'all');t=metric(c,view,'ge20')
            lines.append(f'| {names[view]} | {a["MAE"]:.3f} | {t["MAE"]:.3f} | {t["underprediction"]:.3f} | {t["overprediction"]:.3f} |')
        return '\n'.join(lines)
    p=pd.read_csv(HERE/'common176/long-people.csv');lines=['| 사람 | 실제 TUG | 기존 과소예측 | 역수 과소예측 | 같은 TUG 과소예측 |','|---|---:|---:|---:|---:|']
    for pid,g in p.groupby('id'):
        g=g.set_index('view');lines.append(f'| {pid} | {g.target.iloc[0]:.2f} | {g.loc["Current","underprediction"]:.2f} | {g.loc["Inverse","underprediction"]:.2f} | {g.loc["SameTUG","underprediction"]:.2f} |')
    coefficients={}
    for c,info in protocol['cohorts'].items():
        sels=json.loads((HERE/c/'selections.json').read_text());coefficients[c]={}
        for view in info['views']:
            group=[r for r in sels if r['view']==view];coefs=np.array([joblib.load(HERE/r['file'])[1].coef_ for r in group])
            coefficients[c][view]={'alpha_counts':dict(Counter(str(r['alpha']) for r in group)),'mean_abs_standardized_coefficient':np.mean(abs(coefs),axis=0).tolist(),'mean_signed_coefficient':coefs.mean(0).tolist()}
    save('coefficient-audit.json',coefficients)
    text='''# 보행·회전 개선 후보 비교 — 2026-10-31

**이번 조건에서는 두 후보 모두 뚜렷한 개선을 보이지 않았다.** 237명에서 평균 속도를 역수로 바꿔도 긴 8명의 과소예측은 5.741→5.737초로 사실상 같았다. 같은 TUG의 보행을 쓴 176명 비교에서는 과소예측과 전체 절대오차가 조금 커졌다.

## 요청 조건과 확인한 자료 제약

42.59초인 sub-283을 학습·내부 검증·평가에서 완전히 제외한 기존 237명을 사용했다. 정답은 기존과 같은 사람·방문의 실제 임상 `tft_tug` 시간이다. 20초 이상은 분석용 묶음이며 긴 사람은 8명이다.

같은 TUG 기록에서 기존과 같은 보행 3초를 찾는 입력 점검을 먼저 진행했다. 정의한 보행 구간 판별 기준상 60명은 3초보다 짧았고 1명은 수평 위치 추적 급변이 있어 176명이 통과했다. 긴 8명은 모두 통과했다. 이는 원래 기록에 보행이 없다는 뜻이 아니라, 아래 기준에 맞는 비회전 구간 3초를 확보하지 못했다는 뜻이다.

사용자 선택에 따라 **237명에서는 기존·역수 모델을 비교하고, 176명에서는 세 모델을 모두 다시 학습**했다. 176명에서 얻은 수치를 237명 결과와 직접 비교하지 않는다. 분할 소속은 기존 것을 그대로 제한했으며 새로 섞지 않았다. 176명은 TUG 중앙값 8.38초, 전체 237명은 7.38초여서 선정된 집단의 특성도 다르다.

## 모델 입력과 관찰시간

세 모델 모두 같은 Ridge와 정규화 후보 0.1/1/10/100을 사용했다. 각 바깥 학습 묶음 안에서만 표준화하고 내부 평균 절대오차로 후보를 골랐다. 다섯 사람 분할 × 바깥 다섯 묶음, 각 내부 세 묶음을 그대로 사용했다. 음수 예측만 0초로 처리하고 상한은 두지 않았다.

| 모델 | 입력 네 값 | 이번에 바꾼 부분 |
|---|---|---|
| 기존 단순 Ridge | 별도 10m 보행의 평균 속도·속도 변동·속도 변화, TUG 회전 시간 | 기준 |
| 역수 모델 | 1/평균 보행 속도, 기존 속도 변동·속도 변화, 같은 회전 시간 | 첫 번째 값만 역수로 변경 |
| 같은 TUG 모델 | TUG 안에서 추출한 보행의 평균 속도·속도 변동·속도 변화, 같은 회전 시간 | 속도를 얻는 기록·구간 변경 |

역수 모델은 순간 속도마다 역수를 취한 것이 아니며 평균 속도 하나만 바꿨다. 최저 평균 속도는 0.116m/s로 0이 없어 입력을 잘라내거나 보정하지 않았다. 계수의 방향 제한이나 새 학습 손실은 적용하지 않았다. 따라서 이번 결과는 이 동일 Ridge 조건의 역수 변환에 대한 결과이며 모든 역수·제약 회귀 모델을 검증한 것은 아니다.

각 입력은 **보행 합계 3초 + 원래 회전 구간 D초**다. 237명 비교는 4.13~9.00초, 176명 비교는 4.43~9.00초이며 각 사람의 비교 모델끼리는 정확히 같은 길이다. 회전 구간과 그 시간값은 바꾸지 않았다. 회전에는 기존과 같이 돌아서는 지점의 접근·이탈이 일부 포함된다.

같은 TUG의 보행은 의자 출발점에서 축 방향으로 max(0.3m, 왕복 최대 전진거리의 10%) 이상 떨어진 구간 중 기존 회전 직전·직후에 붙은 두 구간에서 찾았다. 원칙적으로 각각 1.5초를 쓰고, 한쪽이 짧으면 다른 쪽으로 배분해 합계 3초를 맞췄다. 부족한 구간을 복제하거나 시간축을 늘리지 않았다. 수평 골반 속도가 원자료에서 6m/s를 넘는 한 기록은 제외했다. 모든 선택은 움직임과 기존 구간만 사용했으며 임상 TUG 정답이나 모델 성능으로 선택하지 않았다.

30Hz로 속도를 계산하며 떨어진 두 구간 사이를 연결해 가짜 속도 변화를 만들지 않았다. 총 90개 시간 간격의 91개 속도값과 각 구간 내부의 90개 차이를 요약했다. 동일한 하나의 속도 궤적을 나눴다가 합치면 원래 요약과 정확히 같음을 확인했다. 전체 기록은 구간 탐색에 사용했으므로 입력시간이 총 촬영시간과 같다는 뜻은 아니다.

## 전체 237명: 역수 변환의 효과

단위는 초다. 분할별 예측의 오차를 먼저 구한 뒤 사람별·집단별로 평균했다. 절대오차는 과소예측량과 과대예측량의 합이다.

'''+table('all237')+'''

역수 모델의 전체 절대오차 감소는 약 0.002초, 긴 사람 절대오차 감소는 약 0.008초, 과소예측 감소는 약 0.004초다. 긴 사람의 절대오차는 4명 개선·4명 악화였고 분할 평균은 1/5만 개선했다. 과소예측은 3명 개선·4명 악화·1명 변화 없음이었다.

긴 사람 과소예측 차이의 조건부 사람 재표집 95% 범위는 [-0.053, +0.039]초, 절대오차 차이는 [-0.055, +0.035]초로 모두 0을 포함했다. 현재 단순 Ridge보다 유용하게 개선됐다고 볼 근거는 없다.

## 공통 176명: 같은 TUG 보행의 효과

아래 세 모델은 모두 같은 176명으로 다시 학습·평가했다. 237명에서 학습한 기존 모델을 그대로 옮겨 비교하지 않았다.

'''+table('common176')+'''

같은 TUG 속도로 바꾸면 기존 대비 전체 절대오차는 1.151→1.186초, 긴 사람 절대오차는 6.063→6.135초, 과소예측은 5.815→5.969초로 모두 조금 커졌다. 과소예측은 3명 개선·5명 악화, 분할 평균은 1/5만 개선했다. 절대오차는 4명 개선·4명 악화였다. 일부 과대예측은 줄었지만 사용자가 확인하려는 긴 사람 과소예측은 개선되지 않았다.

같은 TUG 모델−기존 모델의 과소예측 차이는 +0.154초, 조건부 95% 범위는 [+0.022, +0.298]초였다. 절대오차 차이는 +0.072초, 범위는 [-0.139, +0.273]초였다. 이는 이미 학습한 모델에서 사람을 재표집한 조건부 범위이며 독립된 집단의 일반화 검증이 아니다.

역수 모델은 이 집단에서 긴 사람 과소예측을 약 0.058초 줄였지만 전체 오차는 소폭 커졌다. 전체 오차와 긴 사람 오차를 함께 줄이는 개선은 확인되지 않았다.

'''+ '\n'.join(lines)+'''

특히 sub-397의 과소예측은 약 13초, sub-406은 약 9초로 세 모델 모두 남았다. 속도의 표현이나 출처를 바꾸는 것만으로 큰 오차가 해결되지 않았다. sub-267의 과소예측이 0인 것은 예측이 정확하다는 뜻이 아니라 기존 모델이 과대예측한 사례였기 때문이다.

## 결과가 의미하는 범위

학습된 표준화 계수의 크기를 확인하면 기존 237명 모델에서는 회전 시간 계수가 평균 약 3.34, 속도 관련 세 계수는 약 0.05~0.13이었다. 이 자료와 모델에서는 회전 시간이 큰 역할을 하고 속도 표현을 바꾼 영향은 작았다는 관찰과 일치한다. 서로 관련된 변수의 계수 크기는 생물학적 중요도나 인과적 기여율이 아니다. 계산은 `coefficient-audit.json`에 남겼다.

같은 TUG라고 해도 이번에는 회전 직전·직후의 3초만 사용했다. 전체 왕복 보행 중 다른 위치의 정지·지체나 준비·기립·착석은 입력에 들어 있지 않다. 특히 긴 8명의 궤적 그림을 확인하면 sub-397은 선택 구간보다 앞에서 의자 축 방향 전진이 오래 정체되는 구간이 보인다. 이 부분은 이번 3초 입력 밖이다. 실제 정지인지 추적 문제인지는 원본 RGB로 확인하지 못했으므로 지연 원인으로 단정하지 않는다. 따라서 같은 TUG의 모든 보행 정보를 사용하는 방법이 효과 없다고 결론내릴 수는 없다. 이번에 정한 짧은 구간과 요약 표현에서는 기존 기준을 넘어서는 이득이 없었다.

공개 시행과 임상 계시의 정확한 cone/line 대응, 원본 영상의 관절 추적 정확도, 구간 경계의 수동 정답은 확인되지 않았다. 골반 궤적과 속도 점검은 원본 RGB에 대한 임상 검증을 대신하지 않는다. 같은 개발자료를 반복 사용한 탐색 비교이며 새 독립 검증 집단은 없다.

## 구현·검산

`scout.py`는 모델 학습 전에 같은 TUG 보행의 길이와 위치 추적 품질을 점검한다. `study.py`는 기존 입력의 평균 속도를 역수로 바꾸거나 같은 TUG 속도 요약으로 대체하고 원래 분할을 유지한다. `train.py`는 각 학습 사람 안에서만 설정을 선택한다. `check_results.py`는 학습 함수를 재사용하지 않고 별도 scikit-learn 객체로 모든 내부 학습과 최종 예측을 재계산한다.

최종 Ridge 125개(237명 50개, 176명 75개)와 내부 검증 학습 1,500회를 독립 재계산해 일치를 확인했다. 237명 기존 모델은 앞선 저장 예측과 정확히 재현됐다. 같은 TUG 속도도 원자료에서 다른 계산 경로로 재추출해 검산했다. 정답·사람 분할·관찰시간·누락/중복·개인 및 분할별 지표·조건부 재표집 범위 확인을 통과했다. 준비 단계의 메모리 tuple과 JSON list 비교 오류를 수정한 뒤 학습했으며, 구간이나 판정 기준은 바꾸지 않았다.

기존 연구 폴더, 패키지, GPU 작업은 변경하지 않았다. 새 폴더에서 로컬 CPU로 실행했다. 재현 순서: `scout.py` → `study.py` → `check_inputs.py` → `train.py` → `assess.py` → `check_results.py` → `finalize.py`. 완료한 준비·학습은 덮어쓰지 않도록 막았다.

근거 파일: `protocol.json`, `scout.json`, `exclusions.json`, 집단별 `metrics.csv`·`long-people.csv`·`partition-metrics.csv`, `input-verification.json`, `results-verification.json`, `comparison.png`, `tug-window-review.png`.
'''
    (HERE/'STATUS.md').write_text(text,encoding='utf8')
    sys.path.insert(0,str(HERE.parents[1]/'control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for ax,c,title in zip(axes,['all237','common176'],['전체 237명: 역수 변환','공통 176명: 세 모델 비교']):
        views=protocol['cohorts'][c]['views'];x=np.arange(len(views));width=.35
        for offset,band,key,color,label in [(-width/2,'all','MAE','#2563eb','전체 평균 절대오차'),(width/2,'ge20','underprediction','#df752d','긴 8명 평균 과소예측')]:
            vals=[metric(c,v,band)[key] for v in views];bars=ax.bar(x+offset,vals,width,color=color,label=label);ax.bar_label(bars,fmt='%.3f',fontsize=9)
        ax.set_xticks(x,[{'Current':'기존','Inverse':'역수','SameTUG':'같은 TUG'}[v] for v in views]);ax.set_ylim(0,8);ax.set_title(title);ax.set_ylabel('초');ax.legend(loc='upper left')
    fig.suptitle('보행 속도 표현·출처 변경: 뚜렷한 개선 없음',fontsize=16)
    fig.text(.5,.015,'42.59초 사례 완전 제외 · 보행 3초 + 같은 회전 구간 · 집단 사이 직접 비교 불가',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(HERE/'comparison.png',dpi=160);plt.close(fig)
    # Inspect all eight long subjects' actual trajectory and selected windows.
    import scout
    rows=[r for r in json.loads((HERE/'common176/manifest.json').read_text()) if r['target']>=20]
    fig,axes=plt.subplots(4,2,figsize=(13,12))
    for ax,m in zip(axes.flat,rows):
        a=scout.read(scout.TURN/'raw'/m['turn_file']);t=a['time'];pos=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);origin=np.median(pos[t<=t[0]+.5],axis=0);progress=(pos-origin)@m['axis']
        rec=next(r for r in json.loads((HERE/'scout.json').read_text()) if r['id']==m['id']);lo=rec['pre'][0]-.5;hi=rec['post'][1]+.5;mask=(t>=lo)&(t<=hi)
        ax.plot(t[mask],progress[mask],color='#334155',lw=1);ax.axvspan(m['turn_start'],m['turn_end'],color='#bfc4cd',alpha=.45)
        for left,right in m['tug_walk_windows']:ax.axvspan(left,right,color='#2563eb',alpha=.3)
        ax.set_title(f'{m["id"]} / 실제 TUG {m["target"]:.2f}초');ax.set_ylabel('의자 축 전진거리 (m)');ax.set_xlabel('원본 기록 시각 (초)');ax.grid(alpha=.2)
    fig.suptitle('긴 8명 입력 점검: 파랑 = 보행 합계 3초 / 회색 = 기존 회전 구간',fontsize=16)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(HERE/'tug-window-review.png',dpi=140);plt.close(fig)
    import sklearn
    save('environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__})
    for name,h in protocol['source_hashes'].items():assert sha(HERE.parent/name)==h
    for p in HERE.glob('*.py'):py_compile.compile(str(p),doraise=True)
    save('artifact-manifest.json',{p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'})
    print('FINALIZED:125 models /1500 inner fits /2 cohorts',flush=True)
if __name__=='__main__':main()
