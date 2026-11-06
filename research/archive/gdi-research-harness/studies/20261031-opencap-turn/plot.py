from common import *
sys.path.append(str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
s=json.loads((HERE/'geometry/summary.json').read_text());p=pd.read_csv(HERE/'geometry/person-results.csv');long=p[p.target>=20]
names=['OldWalk3Speed','Walk3Speed','WalkMatchedSpeed','Walk3TurnTime','Walk3Turn'];labels=['기존 구간 3초','구간 점검 후 3초','같은 시간 직선','회전 시간 추가','회전 정보 추가']
fig,axes=plt.subplots(1,3,figsize=(16,5.5),gridspec_kw={'width_ratios':[1.15,1,1]})
values=[s['models'][n]['all']['MAE'] for n in names];ranges=np.array([s['models'][n]['all']['MAE_range'] for n in names]);ax=axes[0];ax.barh(labels,values,color=['#94a3b8','#94a3b8','#2563eb','#94a3b8','#ea580c']);ax.errorbar(values,np.arange(5),xerr=np.vstack([values-ranges[:,0],ranges[:,1]-values]),fmt='none',color='#334155',capsize=3);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.25);ax.set_xlabel('평균 절대오차 (초, 작을수록 좋음)');ax.set_title('같은 238명 · 15회 반복');ax.grid(axis='x',alpha=.2)
for i,v in enumerate(values):ax.text(v+.04,i,f'{v:.3f}',va='center',fontsize=10)
base=long[long.method=='WalkMatchedSpeed'].set_index('id').sort_index();turn=long[long.method=='Walk3Turn'].set_index('id').loc[base.index]
ax=axes[1];ax.plot([0,45],[0,45],ls='--',color='gray');ax.scatter(base.target,base.prediction,color='#2563eb',label='같은 시간 직선');ax.scatter(turn.target,turn.prediction,color='#ea580c',label='회전 정보 추가');ax.set_xlim(0,45);ax.set_ylim(0,45);ax.set_xlabel('실제 TUG (초)');ax.set_ylabel('15회 평균 예측 (초)');ax.set_title('TUG 20초 이상 9명');ax.legend(fontsize=9);ax.grid(alpha=.2)
ax=axes[2];gain=base.MAE-turn.MAE;colors=['#16a34a' if v>=0 else '#dc2626' for v in gain];ax.barh(base.index,gain,color=colors);ax.axvline(0,color='gray');ax.set_xlabel('직선 오차 - 회전 추가 오차 (초)');ax.set_title('긴 TUG 개인별 오차 감소');ax.grid(axis='x',alpha=.2)
fig.suptitle('방향 전환 추가 비교: 이동 경로로 구간을 찾은 보조 실험',fontsize=15);fig.text(.5,.01,'막대 오차선: 15회 실행의 최소–최대 · 같은 사람은 학습과 평가에 함께 들어가지 않음 · 임상 대체 검증 아님',ha='center',fontsize=10);fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(HERE/'turn-comparison.png',dpi=150);plt.close(fig)
print('PLOT_WRITTEN',flush=True)
