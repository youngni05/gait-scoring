"""Report supporting comparisons and figure from fixed saved predictions."""
import os,sys
import run as r
np,p=r.np,r.p
s=r.m.load_json(r.HERE/'summary.json');d=p.load(r.HERE/'oof_predictions.npz')
rows={v['method']:v for v in s['metrics'] if v['group']=='all695'}
draws=np.random.default_rng(20261029).integers(0,695,(10000,695))
pairs=[('mean_combined_ridge','mean_combined_temporal_ridge','overall'),
       ('mean_leg_ridge','mean_leg_temporal_ridge','overall'),
       ('joint_leg_temporal','joint_combined_temporal','overall'),
       ('previous_leg_gap','gap_angle_ridge','gap'),
       ('previous_leg_gap','gap_no_length_ridge','gap'),
       ('previous_leg_gap','gap_leg_temporal_ridge','gap')]
comparisons=[]
for ref,name,metric in pairs:
    errors=[]
    for method in [ref,name]:
        pred=d['predictions'][r.METHODS.index(method)]
        e=abs(pred-d['y']) if metric=='overall' else abs(pred[:,0]-pred[:,1]-d['y'][:,0]+d['y'][:,1])
        errors.append(p.person_mean(e,d['person_id']))
    delta=errors[0]-errors[1];key='overall_MAE' if metric=='overall' else 'gap_MAE'
    assert abs(delta.mean()-(rows[ref][key]-rows[name][key]))<1e-10
    comparisons.append(dict(reference=ref,method=name,metric=metric,reduction=float(delta.mean()),
        bootstrap95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
train_summary=[]
for method in ['joint_pls32','previous_leg_gap','joint_combined_temporal']:
    for split in ['train','held']:
        group=[v for v in s['train_held'] if v['method']==method and v['split']==split]
        train_summary.append(dict(method=method,split=split,**{key:float(np.mean([v[key] for v in group])) for key in ['overall_MAE','mean_score_MAE','gap_MAE']}))
r.js('descriptive_comparisons.json',dict(exploratory_posthoc=True,selection_changed=False,comparisons=comparisons,train_summary=train_summary))
os.environ['MPLCONFIGDIR']=str(r.HERE/'mplconfig')
sys.path.insert(0,str(r.m.ROOT/'control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
names=['joint_pls32','previous_leg_gap','joint_leg_temporal','joint_combined_temporal']
labels=['기존 PLS','이전 좌우 보완','다리+시간 변화','MB+다리+시간 변화']
ax=axes[0,0];bars=ax.bar(labels,[rows[n]['overall_MAE'] for n in names],color=['#9aa5b1','#6e9298','#5bada0','#087d81'])
ax.bar_label(bars,fmt='%.3f',padding=4);ax.set_ylim(0,9);ax.set_title('전체 GDI 오차: 낮을수록 좋음');ax.set_ylabel('MAE (점)')
ax=axes[0,1]
names2=['gap_position_ridge','gap_angle_ridge','gap_length_ridge','previous_leg_gap','gap_leg_temporal_ridge']
bars=ax.bar(['위치','무릎 각도','영상상 길이','다리 전체','다리+시간 변화'],[rows[n]['gap_MAE'] for n in names2],color=['#a7b4bf','#6e9298','#a7b4bf','#5bada0','#087d81'])
ax.bar_label(bars,fmt='%.3f',padding=4);ax.set_ylim(0,8);ax.set_title('좌우 차이 오차: 특징 종류별 비교');ax.set_ylabel('MAE (점)')
ax=axes[1,0];bands=[40,50,60,70,80,90,100,110]
for name,label,color in [('previous_leg_gap','이전 좌우 보완','#6e9298'),('joint_combined_temporal','MB+다리+시간 변화','#087d81')]:
    vals={v['actual_bin_low']:v for v in s['bands'] if v['method']==name}
    ax.plot(bands,[vals[b]['bias'] for b in bands],marker='o',label=label,color=color)
ax.axhline(0,ls=':',color='gray');ax.set_xlabel('실제 GDI 점수대 시작값');ax.set_ylabel('예측 - 실제 (점)');ax.set_title('양끝 편향은 줄지만 여전히 남음');ax.legend(fontsize=9)
ax=axes[1,1];x=np.arange(2);width=.32
for j,(name,label,color) in enumerate([('previous_leg_gap','이전 좌우 보완','#6e9298'),('joint_combined_temporal','MB+다리+시간 변화','#087d81')]):
    bars=ax.bar(x+(j-.5)*width,[rows[name]['exact_bin_rate']*100,rows[name]['two_or_more_bin_rate']*100],width,label=label,color=color)
    ax.bar_label(bars,fmt='%.2f%%',padding=4)
ax.set_xticks(x,['10점 구간 정확 일치\n(높을수록 좋음)','두 구간 이상 차이\n(낮을수록 좋음)']);ax.set_ylim(0,45);ax.set_ylabel('사람별 비율 평균 (%)');ax.set_title('큰 구간 오류는 줄고, 정확 일치도 줄었음');ax.legend(fontsize=9)
fig.suptitle('695명 CPU 후속 실험 · 시간 변화 특징에서 개선 신호',fontsize=16)
fig.savefig(r.HERE/'comparison.png',dpi=160);plt.close(fig)
print(comparisons)
print(train_summary)
