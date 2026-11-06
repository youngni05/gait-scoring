"""Compute metrics from held-out people, not training logs."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,balanced_accuracy_score,f1_score,confusion_matrix,average_precision_score,roc_auc_score
H=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');B=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103');sys.stdout.reconfigure(encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def metrics(y,p,cl):
    if cl:
        y=y.astype(int);pred=p.argmax(axis=1);cm=confusion_matrix(y,pred,labels=[0,1,2]);den=cm.sum(axis=1)
        return dict(n=len(y),logloss=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()),accuracy=float(np.mean(pred==y)),balanced_accuracy=float(np.mean(np.divide(cm.diagonal(),den,out=np.zeros(3,float),where=den>0))),macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),confusion=cm.tolist(),class_recall=np.divide(cm.diagonal(),den,out=np.zeros(3,float),where=den>0).tolist(),frail_precision=float(cm[2,2]/max(cm[:,2].sum(),1)),frail_average_precision=float(average_precision_score(y==2,p[:,2])) if sum(y==2) else None,frail_auc=float(roc_auc_score(y==2,p[:,2])) if 0<sum(y==2)<len(y) else None)
    e=p-y
    return dict(n=len(y),mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),r2=float(r2_score(y,p)) if len(y)>1 and np.var(y)>0 else None,spearman=float(spearmanr(y,p).statistic) if np.var(y)>0 and np.var(p)>0 else None,median_absolute_error=float(np.median(abs(e))),p90_absolute_error=float(np.quantile(abs(e),.9)),within1=float(np.mean(abs(e)<=1)),errors_over2=float(np.mean(abs(e)>2)),max_absolute_error=float(max(abs(e))))
def paired(y,a,b,cl):
    rng=np.random.default_rng(20261103);draw=rng.integers(0,len(y),size=(10000,len(y)));yy=y.astype(int) if cl else y
    loss_a=-np.log(np.clip(a[np.arange(len(y)),yy],1e-15,1)) if cl else abs(a-y)
    loss_b=-np.log(np.clip(b[np.arange(len(y)),yy],1e-15,1)) if cl else abs(b-y)
    delta=loss_b-loss_a;boot=delta[draw].mean(axis=1);r=dict(metric='logloss' if cl else 'mae',feature_minus_speed=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),conditional_fixed_oof=True)
    if cl:
        da=(b.argmax(axis=1)==yy).astype(float)-(a.argmax(axis=1)==yy).astype(float);bc=[];points=[]
        for c in range(3):
            mask=(yy[draw]==c);bc.append((da[draw]*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1));points.append(da[yy==c].mean())
        ba=np.mean(bc,axis=0);r['balanced_accuracy_feature_minus_speed']=float(np.mean(points));r['balanced_accuracy_ci95']=np.quantile(ba,[.025,.975]).tolist()
    return r
def main():
    rows=read(B/'cohorts.json');allresults={};all_oof=[]
    for target in ['tug','frailty','non_slowness']:
        rr=[r for r in rows if r['status']=='ok' and r[target] is not None];ids=np.array([r['id'] for r in rr]);y=np.array([r[target] for r in rr]);fold=np.array([r['fold'] for r in rr]);cl=target=='frailty';lookup={int(i):j for j,i in enumerate(ids)};predictions={};base=np.zeros((len(y),3) if cl else len(y))
        for outer in range(5):
            tr=fold!=outer;te=fold==outer
            base[te]=np.bincount(y[tr].astype(int),minlength=3)/sum(tr) if cl else np.median(y[tr])
        result=dict(n=len(rr),baseline=metrics(y,base,cl),models={},subgroups={})
        for kind in ['LGBM','TabICLv2','TabPFNv2']:
            kk={}
            for cond in ['speed','features']:
                p=np.zeros_like(base);seen=[]
                for outer in range(5):
                    rec=read(B/'predictions'/f'{target}-{cond}-{outer}-{kind}.json');ctx=np.load(rec['context']);ii=ctx['test_ids'].astype(int).tolist();seen.extend(ii)
                    for i,pred in zip(ii,rec['prediction']):p[lookup[i]]=pred
                assert len(seen)==len(ids) and set(seen)==set(ids);predictions[(kind,cond)]=p;kk[cond]=metrics(y,p,cl)
                for i,actual,pred in zip(ids,y,p):all_oof.append(dict(target=target,kind=kind,condition=cond,id=int(i),actual=float(actual),prediction=pred.tolist() if cl else float(pred)))
            kk['paired_difference']=paired(y,predictions[(kind,'speed')],predictions[(kind,'features')],cl);result['models'][kind]=kk
        groups={'age65plus':np.array([r['age']>=65 for r in rr]),'non_pilot':~np.isin(ids,[1,9,63,100,190,479,482,483,621,622,655,763]),'consistent_items':np.array([r['component_consistent'] for r in rr])}
        if not cl:
            for c in range(3):groups[f'official_frailty_{c}']=np.array([r['frailty']==c for r in rr])
        if target=='tug':
            groups.update({'tug_under8':y<8,'tug_8to12':(y>=8)&(y<12),'tug_12to20':(y>=12)&(y<20),'tug_20plus':y>=20})
        for group,mask in groups.items():
            if not sum(mask):continue
            result['subgroups'][group]={f'{k}/{c}':metrics(y[mask],p[mask],cl) for (k,c),p in predictions.items()}
        result['target_range']=[float(min(y)),float(max(y))];allresults[target]=result
    save(B/'heldout-predictions.json',all_oof);save(H/'comparison.json',dict(protocol_sha256=sha(H/'protocol.json'),cohorts_sha256=sha(B/'cohorts.json'),results=allresults,metrics_code_sha256=sha(Path(__file__))))
    for target,r in allresults.items():
        key='logloss' if target=='frailty' else 'mae';print(target,'n',r['n'],'baseline',r['baseline'][key])
        for k,m in r['models'].items():print(k,m['speed'][key],m['features'][key],m['paired_difference'])
if __name__=='__main__':main()
