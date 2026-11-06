from common import *
sys.path.insert(0,str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def quality_plot():
    d=load_data('TUG_common6');ids=d['ids'];chosen=['sub-001','sub-140','sub-246','sub-266','sub-310','sub-406']
    fig,axs=plt.subplots(3,2,figsize=(12,8),sharex=True)
    for ax,pid in zip(axs.flat,chosen):
        x=d['x'][list(ids).index(pid)];_,q=patterns(x);t=np.arange(x.shape[1])/30
        for side,c,color in [(0,5,'#16669b'),(1,11,'#bc581b')]:
            ax.plot(t,x[c],color=color,label=['Right knee','Left knee'][side],lw=1)
            for a,b in q['spans'][side]:ax.plot(t[[a,b]],x[c,[a,b]],'o',color=color,ms=4)
        ax.set_title(f"{pid}: cycles R/L={q['right_cycles']}/{q['left_cycles']}, flagged={q['bad_fraction']:.1%}")
        ax.set_ylabel('Knee flexion (degrees)');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);axs[-1,0].set_xlabel('Observed time (seconds)');axs[-1,1].set_xlabel('Observed time (seconds)')
    fig.suptitle('Cycle proxies inside the same 6-second walking window\nDots: boundaries of accepted knee-flexion cycles; not annotated foot contacts')
    fig.tight_layout();fig.savefig(HERE/'quality-examples.png',dpi=150);plt.close(fig)

def results_plot():
    p=pd.read_csv(HERE/'person-results.csv');fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    for ax,name,title in zip(axs,['STS_common6','TUG_common6'],['5-times sit-to-stand (179 people)','TUG (195 people)']):
        for view,color in [('speed','#16669b'),('pattern','#bc581b')]:
            g=p[(p.dataset==name)&(p.model=='RF')&(p.view==view)]
            ax.scatter(g.target,g.prediction,s=17,alpha=.65,label=view,color=color)
        limit=float(g.target.max())+2;ax.plot([0,limit],[0,limit],'--',color='gray',lw=1)
        ax.set(xlabel='Measured time (seconds)',ylabel='Mean held-out prediction (seconds)',title=title,xlim=(0,limit),ylim=(0,limit));ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Straight walking only: same 6 seconds and person partitions\nPredictions averaged for display; reported MAE averages errors across runs')
    fig.tight_layout();fig.savefig(HERE/'results.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    quality_plot()
    if (HERE/'person-results.csv').exists():results_plot()
