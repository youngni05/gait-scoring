"""Descriptive control comparison and figure; no fitting or model selection."""
import csv, sys
import run as r
np = r.np
d = r.load(r.HERE/'oof_predictions.npz')
old = r.load(r.HERE/'first_attempt_oof.npz')
for key in d:
    np.testing.assert_array_equal(d[key], old[key])
s = r.m.load_json(r.HERE/'summary.json')
methods = list(d['methods'])
ids, y = d['person_id'], d['y']
draws = np.random.default_rng(20261029).integers(0, 695, (10000, 695))
comparisons = []
for control in ['zero_gap_control', 'train_mean_gap_control']:
    a = d['predictions'][methods.index(control)]
    b = d['predictions'][methods.index('leg_gap_ridge')]
    for metric in ['overall_MAE', 'gap_MAE']:
        def err(p):
            raw = abs(p-y) if metric == 'overall_MAE' else abs((p[:,0]-p[:,1])-(y[:,0]-y[:,1]))
            return r.person_mean(raw, ids)
        delta = err(a)-err(b)
        # Independent summary reconciliation.
        ref = {v['method']: v for v in s['metrics'] if v['group']=='all695'}
        np.testing.assert_allclose(delta.mean(), ref[control][metric]-ref['leg_gap_ridge'][metric], atol=1e-12, rtol=0)
        comparisons.append(dict(control=control, candidate='leg_gap_ridge', metric=metric,
            reduction=float(delta.mean()), bootstrap95=np.percentile(delta[draws].mean(1), [2.5,97.5]).tolist(),
            people_improved=int((delta>1e-12).sum()), people_worsened=int((delta < -1e-12).sum()),
            people_tied=int((abs(delta)<=1e-12).sum())))
r.js('control_comparison.json', dict(exploratory=True, selection_changed=False, first_attempt_predictions_identical=True,
    bootstrap_note='Paired person resampling conditional on saved OOF predictions; excludes selection uncertainty.', comparisons=comparisons))
sys.path.insert(0, str(r.m.ROOT/'control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':font.get_name(), 'axes.unicode_minus':False, 'font.size':11})
fig, axes = plt.subplots(2, 2, figsize=(12,9), constrained_layout=True)
names = ['joint_pls32','zero_gap_control','leg_gap_ridge','combined_gap_ridge']
labels = ['기존 PLS','좌우 차이 0','다리 특징','MB+다리 특징']
colors = ['#8b98a8','#b8c1cb','#167c80','#6a9c92']
rows = {v['method']:v for v in s['metrics'] if v['group']=='all695'}
for ax,metric,title in [(axes[0,0],'overall_MAE','전체 GDI 오차 (낮을수록 좋음)'),(axes[0,1],'gap_MAE','좌우 차이 오차 (낮을수록 좋음)')]:
    values=[rows[n][metric] for n in names]
    bars=ax.bar(labels,values,color=colors)
    ax.bar_label(bars,fmt='%.3f',padding=4)
    ax.set_ylim(0,9);ax.set_ylabel('MAE (점)');ax.set_title(title)
ax=axes[1,0]
values=[rows['joint_pls32']['actual_gap_magnitude'],rows['joint_pls32']['predicted_gap_magnitude'],rows['leg_gap_ridge']['predicted_gap_magnitude']]
bars=ax.bar(['실제 차이','기존 PLS','다리 특징'],values,color=['#d09b43',colors[0],colors[2]])
ax.bar_label(bars,fmt='%.2f',padding=4);ax.set_ylim(0,8.5);ax.set_ylabel('평균 |왼쪽 - 오른쪽| (점)');ax.set_title('좌우 차이의 크기: 여전히 작게 예측')
ax=axes[1,1]
truth=r.person_mean(y[:,0]-y[:,1],ids)
for name,label,color in [('joint_pls32','기존 PLS',colors[0]),('leg_gap_ridge','다리 특징',colors[2])]:
    p=d['predictions'][methods.index(name)]
    gap=r.person_mean(p[:,0]-p[:,1],ids)
    ax.scatter(truth,gap,s=10,alpha=.4,label=label,color=color)
ax.plot([-45,45],[-45,45],':',color='#777777',label='정확한 예측')
ax.set_xlim(-45,45);ax.set_ylim(-45,45);ax.set_xlabel('실제 왼쪽 - 오른쪽 (점)');ax.set_ylabel('예측 왼쪽 - 오른쪽 (점)');ax.set_title('사람별 좌우 차이');ax.legend(fontsize=9)
fig.suptitle('695명 CPU 실험 · 평균 점수는 유지하고 좌우 차이만 변경',fontsize=16)
fig.savefig(r.HERE/'comparison.png',dpi=160)
plt.close(fig)
print(comparisons)
