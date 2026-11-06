from study import *
sys.path.insert(0,str(HERE.parents[1]/'control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    rows=json.loads((HERE/'include283/manifest.json').read_text());ids=['sub-283','sub-397','sub-310','sub-288']
    fig,axes=plt.subplots(4,1,figsize=(12,13),layout='constrained')
    for ax,id in zip(axes,ids):
        m=next(m for m in rows if m['id']==id);a=read(TURN/'raw'/m['turn_file']);r=m['comparison_budgets']['20'];phase=m['phase']
        ax.plot(a['time'],a['pelvis_ty'],color='#34495e',lw=1,label='Pelvis height')
        for window,color,label in [(r['original']['window'],'#8da0cb','Original gait window'),(phase['extended'],'#fc8d62','Full prep/rise candidate'),(r['rise_clip'],'#66c2a5','Observed prep/rise clip')]:
            ax.axvspan(*window,color=color,alpha=.22,label=label)
        lo,hi=ax.get_ylim();yr=hi-lo
        for w in [r['rise_clip'],r['reallocated_gait']['window']]:ax.plot(w,[lo+yr*.08]*2,color='#1b7837',lw=5)
        ax.axvline(phase['core'][0],color='#e66101',ls=':',lw=1.5,label='Core rise start')
        ax.set_title(f'{id}: clinical TUG {m["target"]:.2f}s | equal observed total {r["total_observed"]:.2f}s | green bars = new clips')
        ax.set_ylabel('Pelvis height (m)');ax.set_xlabel('Recording time (s)');ax.grid(alpha=.2)
    axes[0].legend(loc='upper left',fontsize=8,ncol=3)
    fig.suptitle('Automatic boundaries; clinical command/timed trial not confirmed',fontsize=13)
    fig.savefig(HERE/'phase-review.png',dpi=150);plt.close(fig)
    print('PHASE PLOT SAVED',flush=True)
if __name__=='__main__':main()
