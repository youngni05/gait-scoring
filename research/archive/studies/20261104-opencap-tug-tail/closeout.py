from pathlib import Path
import json, hashlib, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path('D:/Projects/D_P/studies/20261104-opencap-tug-tail')
HEAVY=Path('D:/gait-research-data/D_P/opencap-tug-tail-20261104')
SOURCE=Path('D:/gait-research-data/D_P/opencap-camera-tug-20261104')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank(x):
    u,idx,c=np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(c)-(c-1)/2)[idx]
def main():
    result=json.loads((HERE/'results.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['passed'] and sha(HERE/'results.json')==v['results_sha256']
    d=np.load(SOURCE/'features.npz');X=d['X'];names=d['names'].tolist();y=d['y'];ids=d['ids']
    correlations={}
    for n in ['ankle_separation_range','L_knee_flexion_2d_std','R_knee_flexion_2d_std','extrema_cadence','extrema_step_time_cv','trunk_lean_std']:
        a=X[:,names.index(n)];ok=np.isfinite(a)
        correlations[n]=float(np.corrcoef(rank(X[ok,0]),rank(a[ok]))[0,1])
    rows=json.loads((HEAVY/'predictions.json').read_text())
    blind=[{'repeat':r['repeat'],'method':r['name'],'flag':r['flag']} for r in rows if r['id']=='sub-310' and r['name'] in ['all/predgate10','all/logitgate05','all/treegate03']]
    extra={'speed_feature_Spearman':correlations,'sub310_blindspot_routing':blind,
       'interpretation':'Correlations describe overlap with speed, not proof of causal redundancy. Sub310 is a descriptive failure case, not a rule to exclude.'}
    (HERE/'additional-diagnostics.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')
    font_manager.findfont('Malgun Gothic',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    z=np.load(HEAVY/'verified-predictions.npz');old=z['previous__speed_plus_landmarks'].mean(0)
    high=y>=20
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.8),constrained_layout=True)
    for g,c,label in [(~high,'#7f9eaa','실제 TUG <20초'),(high,'#ce5a46','실제 TUG ≥20초 (9명)')]:
        ax[0].scatter(X[g,0],y[g],c=c,s=24 if g is not high else 40,alpha=.75,label=label)
        ax[1].scatter(y[g],old[g],c=c,s=24,alpha=.75)
    ax[0].set(xlabel='3초 직선 보행 속도 (m/s)',ylabel='실제 TUG (초)',title='직선 보행이 비슷해도 TUG는 크게 다르다')
    i=np.flatnonzero(ids=='sub-283')[0]
    ax[0].annotate('0.62m/s, TUG 42.59초',xy=(X[i,0],y[i]),xytext=(.95,38),arrowprops={'arrowstyle':'->','color':'#444'})
    ax[0].legend(loc='upper right',fontsize=8)
    ax[1].plot([0,46],[0,46],'--',color='#555',lw=1,label='실제 = 예측')
    ax[1].set(xlabel='실제 TUG (초)',ylabel='기존 모델의 예측 TUG (초)',title='높은 TUG를 낮게 예측하는 현상',xlim=(3,46),ylim=(3,46))
    ax[1].annotate('42.59초 → 평균 12.14초',xy=(y[i],old[i]),xytext=(18,28),arrowprops={'arrowstyle':'->','color':'#444'})
    for a in ax:a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
    fig.suptitle('OpenCap 297명 · 사람 분리 5회 반복 · 다중 카메라 좌표의 측면 투영',fontsize=12)
    fig.savefig(HERE/'tail-analysis.png',dpi=180);plt.close(fig)
    lines=['OpenCap 높은 TUG 오차 분석 및 개선 비교 — 실행 결과 기록','',
      '분석 대상: 297명, TUG<12초264명 /12~20초24명 />=20초9명. 기존과 같은25개 바깥 사람 분할.',
      '새 정보 없이 모델을 변경한 실험이다. 추가 TUG 동작9명 파일은 연결 가능성 점검에만 사용했다.',
      '모든 수치는 사람별5회 반복 오차를 먼저 평균한 뒤 집계했다. 새로운 외부 검증이 아니다.','',
      '방식 | 전체 MAE | <12초 MAE | >=20초 MAE']
    labels={'previous/speed_plus_landmarks':'기존 모델','all/base':'고정 트리 + 역속도 입력 기준','all/predgate12':'예측12초 이상 전용 모델 재계산',
      'all/lds1':'희귀 점수 가중 강화','two_stage_constrained':'학습 내부에서 제한적으로 두 단계 선택'}
    for n,label in labels.items():
        g=result['summary'][n]['groups'];lines.append(f"{label} | {g['all']['MAE_repeated']:.4f} | {g['under12']['MAE_repeated']:.4f} | {g['20plus']['MAE_repeated']:.4f}")
    lines+=['','확인한 결과: 전용 모델 및 희귀 점수 가중은 높은 TUG 오차를 줄였지만 전체/낮은TUG 오차가 늘었다.',
      '제한적 두 단계 선택: 같은 새 기준모델 대비 >=20초 MAE 차이 -0.6723초 (95% -1.2611~-0.1694), 전체 +0.0263초 (95% -0.0132~+0.0697).',
      '두 단계는25학습 중4학습에서만 선택됐고21학습은 기준 모델로 돌아갔다. 안정적 임상 개선으로 해석하지 않는다.',
      '실제 TUG42.59초 사례는 기존 평균12.14초에서 제한적 두 단계15.44초로 바뀌었으나 큰 오차가 남았다.',
      '시작 신호가 없어 TUG 기록의 좌석 근처 체류 시간을 실제 일어나기 지연으로 확정할 수 없다.',
      '실제 휴대폰 한 대/MediaPipe 정확도, 일반 고령자 정확도, 독립 외부 검증은 미확인이다.',
      '검산: 200묶음/2,000저장모델 재추론, 학습 자료만의 보정/가중치, 사람 분리, 내부 선택 통과.']
    (HERE/'STATUS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    artifacts=['experiment.py','verify.py','protocol.json','results.json','verification.json','diagnosis.json','additional-task-linkage-pilot.json','additional-diagnostics.json','comparison.csv','STATUS.txt','tail-analysis.png','bundle-hashes.json']
    (HERE/'artifact-manifest.json').write_text(json.dumps({n:sha(HERE/n) for n in artifacts},indent=2),encoding='utf-8')
    print('CLOSEOUT_RECORDED_AND_HASHED',flush=True)
if __name__=='__main__':main()
