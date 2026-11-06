from common import *
sys.path.insert(0,str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    q=pd.read_csv(HERE/'quality-audit.csv');q=q[q.dataset=='TUG_common6']
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for stable,color,label in [(False,'#bc581b','Flagged'),(True,'#16669b','Stable proxy')]:
        z=q[q.stable==stable]
        axs[0].scatter(z.bad_fraction,z.cycle_seconds_spread,s=22,alpha=.6,c=color,label=label)
    axs[0].set(xlabel='Fraction flagged for abrupt angle changes',ylabel='Cycle duration range across 9 settings (s)',title='Input consistency, not manual tracking truth');axs[0].legend();axs[0].grid(alpha=.2)
    data=load_data('TUG_common6')
    # First stable and highest disagreement examples chosen by inputs, never by errors.
    stable_id=q[q.stable].id.iloc[0];fragile_id=q.sort_values('cycle_seconds_spread',ascending=False).id.iloc[0]
    for pid,color in [(stable_id,'#16669b'),(fragile_id,'#bc581b')]:
        x=data['x'][list(data['ids']).index(pid)]
        vals=[patterns(x,w,p)[0][0] for w,p in VARIANTS]
        axs[1].plot(range(9),vals,'o-',color=color,label=pid)
    axs[1].set(xlabel='Fixed extraction setting',ylabel='Estimated cycle duration (s)',title='Sensitivity to smoothing and peak prominence');axs[1].legend();axs[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'consistency-audit.png',dpi=160);plt.close(fig)
    if not (HERE/'metrics.csv').exists():return
    comparisons=json.loads((HERE/'comparisons.json').read_text());fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ax,name,title in zip(axs,['STS_common6','TUG_common6'],['5-times sit-to-stand','TUG']):
        for group,color,offset in [('all','#16669b',-.1),('stable','#bc581b',.1)]:
            z=[next(c for c in comparisons if c['dataset']==name and c['model']=='RF' and c['group']==group and c['new']==new and c['base']==base)
               for new,base in [('quality','speed'),('pattern','quality'),('robust','quality')]]
            vals=np.array([c['mae_delta'] for c in z]);lo=np.array([c['ci95'][0] for c in z]);hi=np.array([c['ci95'][1] for c in z])
            ax.errorbar(np.arange(3)+offset,vals,yerr=np.array([vals-lo,hi-vals]),fmt='o',capsize=3,color=color,label=f"{group} (n={z[0]['n']})")
        ax.axhline(0,color='gray',ls='--',lw=1)
        ax.set_xticks(range(3),['Quality vs speed','Pattern vs quality','Robust vs quality']);ax.tick_params(axis='x',labelsize=8)
        ax.set(ylabel='Paired MAE change (seconds; negative = better)',title=title);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same people and 6-second windows within each comparison\nConditional 95% person-bootstrap intervals; no independent external test')
    fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
