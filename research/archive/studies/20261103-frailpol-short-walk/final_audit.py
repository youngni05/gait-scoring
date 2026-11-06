from pathlib import Path
import json,hashlib,sys,warnings
import numpy as np
import openpyxl
from sklearn.metrics import mean_absolute_error,mean_squared_error,log_loss,balanced_accuracy_score,f1_score
from sklearn.model_selection import StratifiedKFold
H=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');B=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103');sys.stdout.reconfigure(encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    v=read(H/'verification.json');s=read(H/'signal-verification.json');assert v['status']==s['status']=='passed';assert v['models']==90 and s['people']==596
    receipt=read(H/'run-receipt.json');assert len(list((B/'contexts').glob('*.npz')))==30 and len(list((B/'models').glob('*.txt')))==30
    for n,h in receipt['predictions'].items():assert sha(B/'predictions'/n)==h
    comparison=read(H/'comparison.json');rows=read(B/'cohorts.json');byid={r['id']:r for r in rows}
    workbook=openpyxl.load_workbook(B/'raw/participants.xlsx',read_only=True,data_only=True);raw=list(workbook.active.values);headers=[str(v).strip() for v in raw[0]];source={int(r[0]):dict(zip(headers,r)) for r in raw[1:] if isinstance(r[0],(int,float))}
    items=['Weight loss (0/1)','Weakness (0/1)','Slowness (0/1)','Physical Activity (0/1)','Muscular strength (0/1)']
    for r in rows:
        src=source[r['id']];assert r['frailty']==int(src['Targets']) and r['age']==float(src['Age']);t=src['TUG time (s)'];assert r['tug']==t or (r['tug'] is None and t is None)
        if r['non_slowness'] is not None:assert sum(src[k] for k in items)==src['FI'] and r['non_slowness']==sum(src[k] for k in items if k!='Slowness (0/1)')
    ids=np.array(sorted(byid));ys=np.array([byid[int(i)]['frailty'] for i in ids])
    for fold,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261103).split(ids,ys)):assert all(byid[int(ids[i])]['fold']==fold for i in te)
    pp=read(B/'heldout-predictions.json');metrics_checked=0;intervals=[];probability_checks=[]
    for target in ['tug','frailty','non_slowness']:
        cl=target=='frailty'
        for kind in ['LGBM','TabICLv2','TabPFNv2']:
            paired=[];pair_ids=[]
            for cond in ['speed','features']:
                rr=sorted([r for r in pp if r['target']==target and r['kind']==kind and r['condition']==cond],key=lambda r:r['id']);ii=[r['id'] for r in rr];y=np.array([r['actual'] for r in rr]);p=np.array([r['prediction'] for r in rr]);pair_ids.append(ii);assert y.tolist()==[byid[i][target] for i in ii];m=comparison['results'][target]['models'][kind][cond]
                if cl:
                    # JSON float64 preserves original float32 outputs; keep raw values and audit rounding.
                    with warnings.catch_warnings(record=True) as ws:value=log_loss(y,p,labels=[0,1,2])
                    deviation=float(max(abs(p.sum(axis=1)-1)));assert deviation<1e-5
                    normalized=log_loss(y,p/p.sum(axis=1)[:,None],labels=[0,1,2]);probability_checks.append(dict(kind=kind,condition=cond,max_row_sum_deviation=deviation,normalizing_logloss_difference=float(normalized-value),warnings=[str(w.message) for w in ws]))
                    assert abs(normalized-value)<1e-6;assert abs(value-m['logloss'])<1e-10;assert abs(balanced_accuracy_score(y,p.argmax(axis=1))-m['balanced_accuracy'])<1e-10;assert abs(f1_score(y,p.argmax(axis=1),average='macro')-m['macro_f1'])<1e-10;paired.append(-np.log(p[np.arange(len(y)),y.astype(int)]))
                else:
                    assert abs(mean_absolute_error(y,p)-m['mae'])<1e-10;assert abs(np.sqrt(mean_squared_error(y,p))-m['rmse'])<1e-10;paired.append(abs(p-y))
                metrics_checked+=1
            assert pair_ids[0]==pair_ids[1];d=paired[1]-paired[0];rng=np.random.default_rng(20261103);boot=np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(10000)]);ci=np.quantile(boot,[.025,.975]);saved=comparison['results'][target]['models'][kind]['paired_difference'];assert np.allclose(ci,saved['ci95'],atol=1e-12,rtol=1e-12);assert abs(d.mean()-saved['feature_minus_speed'])<1e-12;intervals.append(dict(target=target,kind=kind,ci95=ci.tolist()))
    result=dict(status='passed',provider_labels_checked=len(rows),conditions_checked=metrics_checked,paired_intervals_checked=len(intervals),saved_models_replayed=v['models'],raw_feature_replays=s['people'],outside_window_checks=s['perturbation_people'],final_audit_sha256=sha(Path(__file__)),comparison_sha256=sha(H/'comparison.json'),intervals=intervals,probability_checks=probability_checks)
    (H/'final-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ['intervals','probability_checks']},indent=2))
if __name__=='__main__':main()
