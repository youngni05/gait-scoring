"""Visualize verified preset cohorts; no fitting or new statistical choices."""
from prepare_inputs import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(CASE));labels=module('existing_labels',CASE/'report.py').LABEL.copy();labels['single_CNN']='Single CNN (first window)'
def main():
 seal=read(HERE/'plot-code-seal.json');assert sha(__file__)==seal['script_sha256'];r=read(HERE/'results.json');v=read(HERE/'verification.json');assert v['passed'] and v['results_sha256']==sha(HERE/'results.json');names=r['models']
 fig,axes=plt.subplots(1,2,figsize=(14,7.4),layout='constrained');ys=np.arange(len(names));groups=r['groups']
 for qu,offset,color,label in [('standard_window_gate_only',-.12,'#2457a7','Missing coordinates <=25%'),('high_missing_exploratory',.12,'#db8b23','High-missing exploratory')]:
  g=groups['primary_remaining__'+qu];mae=[np.nan if g[n]['all']['mae'] is None else g[n]['all']['mae'] for n in names];axes[0].scatter(mae,ys+offset,s=34,color=color,label=label)
 axes[0].set_yticks(ys,[labels[n] for n in names],fontsize=8);axes[0].invert_yaxis();axes[0].set(xlabel='Person-equal MAE (GDI points)',title='Remaining identified people, separate quality groups');axes[0].legend(loc='best',fontsize=8);axes[0].grid(axis='x',alpha=.2)
 with np.load(BIG/'scored.npz',allow_pickle=False) as z:
  unlinked=z['cohort']=='unlinked';representatives=['fusion','plus_ranksim','plus_HuberRank1','plus_MAERank1'];data=[]
  for n in representatives:
   scores=z[n][unlinked].mean(1);data.append(scores[np.isfinite(scores)])
 axes[1].boxplot(data,tick_labels=['Fusion','+ RankSim','+ HuberRank','+ MAERank'],showfliers=True,orientation='vertical',flierprops=dict(markersize=2,alpha=.4));axes[1].set(ylabel='Predicted mean of left/right GDI',title='Unlinked clips: predictions only, no accuracy claim');axes[1].grid(axis='y',alpha=.2)
 fig.savefig(HERE/'remaining-data-comparison.png',dpi=150);plt.close(fig);save(HERE/'plot-receipt.json',dict(passed=True,script_sha256=sha(__file__),results_sha256=sha(HERE/'results.json'),verification_sha256=sha(HERE/'verification.json'),figure_sha256=sha(HERE/'remaining-data-comparison.png'),note='Quality groups differ in people and truth distribution; not a causal comparison. Unknown-identity boxes are predictions, not measured accuracy.'))
 print('Verified remaining-data figure saved',flush=True)
if __name__=='__main__':main()
