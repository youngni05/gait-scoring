"""Reuse frozen held-out predictions; no new model or probability calibration."""
from common import *
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix

def main():
    src=HERE.parent/'20261031-opencap-walk-quality';rows=[];meta={}
    for outcome in ['STS','TUG']:
        name=outcome+'_common6';p=src/name/'predictions.csv';df=pd.read_csv(p)
        inputdata=dict(np.load(src/name/'input.npz'));ids=inputdata['ids'];y=inputdata['y'];n=len(ids)
        assert 'sub-283' not in ids and len(set(ids))==n
        meta[outcome]={'people':n,'ge20_people':int((y>=20).sum()),'prevalence':float((y>=20).mean()),'prediction_sha256':sha(p),
                       'note':'20s inherited descriptive tail threshold, not clinical diagnosis; repeated partitions are not independent people.'}
        for (model,view,seed,rep),g in df[df.view.isin(['speed','quality','pattern','robust'])].groupby(['model','view','seed','repeat']):
            assert g.id.nunique()==len(g)==n
            label=g.target.to_numpy()>=20;pred=g.prediction.to_numpy();decision=pred>=20
            tn,fp,fn,tp=confusion_matrix(label,decision,labels=[False,True]).ravel()
            rows.append(dict(outcome=outcome,model=model,view=view,seed=int(seed),repeat=int(rep),people=n,positives=int(label.sum()),
                auroc=float(roc_auc_score(label,pred)),average_precision=float(average_precision_score(label,pred)),
                sensitivity_at_predicted20=float(tp/(tp+fn)),specificity_at_predicted20=float(tn/(tn+fp)),
                tp=int(tp),fn=int(fn),fp=int(fp),tn=int(tn)))
    table=pd.DataFrame(rows);table.to_csv(HERE/'current-heldout-ranking.csv',index=False)
    aggregate=[]
    for key,g in table.groupby(['outcome','model','view']):
        r=dict(zip(['outcome','model','view'],key));r['runs']=len(g)
        for metric in ['auroc','average_precision','sensitivity_at_predicted20','specificity_at_predicted20','tp','fn','fp']:
            r[metric+'_mean']=float(g[metric].mean());r[metric+'_min']=float(g[metric].min());r[metric+'_max']=float(g[metric].max())
        aggregate.append(r)
    pd.DataFrame(aggregate).to_csv(HERE/'current-ranking-summary.csv',index=False)
    save('current-probe.json',{'cohorts':meta,'scope':'descriptive reinterpretation of existing held-out regression rankings; not a trained/calibrated risk classifier, probability interval or external validation','new_models':0})
    print(pd.DataFrame(aggregate).query("model=='RF' and view in ['speed','quality']")[[c for c in pd.DataFrame(aggregate).columns if c in ['outcome','view','auroc_mean','average_precision_mean','sensitivity_at_predicted20_mean','tp_mean','fp_mean']]].to_string(index=False))
if __name__=='__main__':main()
