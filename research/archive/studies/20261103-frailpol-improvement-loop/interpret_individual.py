"""Descriptive held-out person errors/ranking; no fitting or model selection."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import spearmanr,pearsonr
from sklearn.metrics import r2_score
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');D=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104/data.npz')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=np.load(D);ids=d['ids'];y=d['y'];fold=d['fold'];speed=d['F'][:,0];assert len(ids)==594 and sum(y>=20)==9
source=B/'round12/heldout.json';verification=read(H/'round12/verification.json');assert verification['status']=='passed' and verification['heldout_sha256']==sha(source)
rows=read(source);out={};pred={};i,j=np.triu_indices(594,1);same=fold[i]==fold[j];i=i[same];j=j[same];gap=np.abs(y[i]-y[j]);same_speed=np.abs(np.log(speed[i])-np.log(speed[j]))<=np.log(1.05)
for c in ['speed/config1/guarded','speed_duration/config1/guarded','combined/config1/guarded','combined/config3/guarded']:
    lookup={int(r['id']):r for r in rows if r['condition']==c};assert set(lookup)==set(ids) and all(lookup[int(k)]['actual']==v for k,v in zip(ids,y));p=np.array([lookup[int(k)]['prediction'] for k in ids]);pred[c]=p;e=np.abs(p-y);correct=(p[i]-p[j])*(y[i]-y[j])>0;pair={}
    for lo,hi in [(.5,1),(1,2),(2,3),(3,5),(5,np.inf)]:
        mask=(gap>=lo)&(gap<hi);pair[f'{lo}to{hi}']=dict(pairs=int(mask.sum()),correct_order_fraction=float(correct[mask].mean()))
    mask=same_speed&(gap>=2)
    out[c]=dict(mae=float(e.mean()),median_absolute_error=float(np.median(e)),rmse=float(np.sqrt(np.mean(e**2))),r2=float(r2_score(y,p)),pearson=float(pearsonr(y,p).statistic),spearman=float(spearmanr(y,p).statistic),error_within={str(t):dict(n=int((e<=t).sum()),fraction=float((e<=t).mean())) for t in [1,2,3]},max_absolute_error=float(e.max()),bands={label:dict(n=int(m.sum()),mae=float(e[m].mean()),bias=float((p-y)[m].mean()),spearman=float(spearmanr(y[m],p[m]).statistic)) for label,m in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]},same_test_fold_pair_order_by_true_TUG_gap=pair,estimated_speed_within5percent_true_TUG_gap2plus_same_fold=dict(pairs=int(mask.sum()),correct_order_fraction=float(correct[mask].mean())))
base=np.abs(pred['speed/config1/guarded']-y);new=np.abs(pred['combined/config1/guarded']-y);improve=base-new
result=dict(status='descriptive_existing_heldout_predictions_only',people=594,source_heldout_sha256=sha(source),source_verification_sha256=sha(H/'round12/verification.json'),data_sha256=sha(D),conditions=out,fullA_vs_speedA=dict(people_improved=int((improve>1e-12).sum()),people_worsened=int((improve<-1e-12).sum()),ties=int((np.abs(improve)<=1e-12).sum()),median_absolute_error_reduction=float(np.median(improve)),mean_absolute_error_reduction=float(improve.mean())),limits='No new training or thresholds used for model selection. Crossvalidated predictions are reused development evidence. Pair comparisons restricted to same outer test fold, hence same held-out model; pairs share people and are not independent observations. Gap bins and5percent speed matching are descriptive choices, not clinical cutoffs. Only1functional label/person: cannot measure longitudinal responsiveness, repeatability, or individual minimum detectable change. Estimated speed is not gold ground truth. No camera/SPPB validation.',code_sha256=sha(Path(__file__)))
(H/'round12/individual-interpretation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(result,indent=2))
