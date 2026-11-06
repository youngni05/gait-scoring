"""Exploratory same-125 ordinal regression and risk-gated regression."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,sys,time,warnings
import numpy as np,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import StratifiedKFold
import prepare as p
from review import metrics,paired
warnings.filterwarnings('ignore',message='X does not have valid feature names')
H=p.H;B=p.B
ORD=[{'C':c,'stat':stat} for c in [.01,.1,1] for stat in ['median','mean']]
GATE=[{'C':c,'alpha':a} for c in [.1,1,10] for a in [10,100]]

def freeze():
    assert not(H/'round3-protocol.json').exists()
    p.save(H/'round3-protocol.json',{'code_sha256':p.sha(__file__),'round1_results_sha256':p.sha(H/'results.json'),'input_sha256':p.sha(B/'input.npz'),
      'reason':'Round1upper-tailunderprediction;testorderedscoremodellingandriskgatingwithoutclinicalinputs. Exploratoryreused5folddevelopment,notexternalconfirmation.',
      'input':'All125same57same-window2Dfeatures,originalpersonfolds;noageorclinicalcomponents.',
      'ordinal':'Thresholds1to31;train-onlystandardization;L2logisticC=.01/.1/1;constantifoneclass;enforcenonincreasingcumulativerisk;predictconditionalmedianormean;selectall6byinner3MAE.',
      'gate':'Trainlogisticgateforscore>=6,C=.1/1/10;twoRidgeexpertsalpha10/100onlow/hightrainpeople;softprobabilitymixture;selectall6byinner3MAE. Testlabelsneverchooseexpert.',
      'selection':'Sameinner3personfolds,seed20261105;outertestsealeduntilall5complete;nochangeofprimarytarget,goalsorcomparison.',
      'failure_rule':'Do not keep iterating against reusedouterresultsuntilagoodnumberappears. Ifgoalsmissed,reportfailureanddatavaliditylimits;noalternativeendpointasprimary.'})

def fit(X,y,cfg,arm):
    scaler=StandardScaler().fit(X);z=scaler.transform(X)
    if arm=='ordinal':
        models=[]
        for threshold in range(1,32):
            target=(y>=threshold).astype(int)
            m=float(target[0]) if len(np.unique(target))==1 else LogisticRegression(C=cfg['C'],max_iter=500,solver='lbfgs',random_state=20261105).fit(z,target)
            models.append(m)
        return {'arm':arm,'config':cfg,'scaler':scaler,'models':models,'ntrain':len(y)}
    group=(y>=6).astype(int);gate=LogisticRegression(C=cfg['C'],max_iter=500,random_state=20261105).fit(z,group)
    experts=[Ridge(alpha=cfg['alpha']).fit(z[group==k],y[group==k]) for k in [0,1]]
    return {'arm':arm,'config':cfg,'scaler':scaler,'gate':gate,'experts':experts,'ntrain':len(y),'group_counts':[int(sum(group==k)) for k in [0,1]]}

def predict(m,X):
    z=m['scaler'].transform(X)
    if m['arm']=='ordinal':
        probs=np.column_stack([np.full(len(X),item) if isinstance(item,float) else item.predict_proba(z)[:,1] for item in m['models']])
        probs=np.minimum.accumulate(probs,axis=1)
        return (probs>=.5).sum(1).astype(float) if m['config']['stat']=='median' else probs.sum(1)
    prob=m['gate'].predict_proba(z)[:,1];lo=m['experts'][0].predict(z);hi=m['experts'][1].predict(z)
    return np.maximum(0,(1-prob)*lo+prob*hi)

def train():
    protocol=json.loads((H/'round3-protocol.json').read_text());assert protocol['code_sha256']==p.sha(__file__)
    d=np.load(B/'input.npz');folder=B/'round3-models';folder.mkdir(exist_ok=True);records=[];start=time.monotonic()
    for f in range(5):
      tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261105).split(tr,np.digitize(d['y'][tr],[6,14])))
      for arm,configs in [('ordinal',ORD),('gate',GATE)]:
        A=np.empty((len(tr),len(configs)));inner=[]
        for k,cfg in enumerate(configs):
          for j,(it,iv) in enumerate(splits):
            ti=tr[it];vi=tr[iv];m=fit(d['X'][ti],d['y'][ti],cfg,arm);A[iv,k]=predict(m,d['X'][vi]);path=folder/f'f{f}-{arm}-k{k}-i{j}.joblib';joblib.dump(m,path,compress=3)
            inner.append({'candidate':k,'inner':j,'train':ti.tolist(),'test':vi.tolist(),'path':str(path),'sha256':p.sha(path)})
        error=abs(A-d['y'][tr,None]).mean(0);k=int(np.argmin(error));m=fit(d['X'][tr],d['y'][tr],configs[k],arm);pr=predict(m,d['X'][te]);path=folder/f'f{f}-{arm}-final.joblib';joblib.dump(m,path,compress=3);ctx=folder/f'f{f}-{arm}-context.npz';np.savez_compressed(ctx,A=A,train=tr,test=te,pred=pr)
        records.append({'fold':f,'arm':arm,'selected':k,'inner_mae':error.tolist(),'inner':inner,'path':str(path),'sha256':p.sha(path),'context':str(ctx),'context_sha256':p.sha(ctx)})
        p.save(H/'round3-receipt.json',{'records':records,'code_sha256':p.sha(__file__),'input_sha256':p.sha(B/'input.npz'),'seconds':time.monotonic()-start});print('ROUND3_FIT',f,arm,round(time.monotonic()-start,1),flush=True)

def replay_prediction(m,X):
    z=(X-m['scaler'].mean_)/m['scaler'].scale_
    def probability(model):
        if isinstance(model,float):return np.full(len(X),model)
        zz=np.clip(z@model.coef_[0]+model.intercept_[0],-700,700);return 1/(1+np.exp(-zz))
    if m['arm']=='ordinal':
        P=np.minimum.accumulate(np.column_stack([probability(item) for item in m['models']]),axis=1)
        return (P>=.5).sum(1).astype(float) if m['config']['stat']=='median' else P.sum(1)
    prob=probability(m['gate']);lo=z@m['experts'][0].coef_+m['experts'][0].intercept_;hi=z@m['experts'][1].coef_+m['experts'][1].intercept_
    return np.maximum(0,lo*(1-prob)+hi*prob)

def review():
    d=np.load(B/'input.npz');old=np.load(B/'verified-predictions.npz');receipt=json.loads((H/'round3-receipt.json').read_text());protocol=json.loads((H/'round3-protocol.json').read_text())
    assert protocol['code_sha256']==receipt['code_sha256']==p.sha(__file__) and receipt['input_sha256']==protocol['input_sha256']==p.sha(B/'input.npz')
    assert protocol['round1_results_sha256']==p.sha(H/'results.json') and len(receipt['records'])==10
    pred={a:np.full(125,np.nan) for a in ['ordinal','gate']};count=0;maxdiff=0
    for row in receipt['records']:
        f=row['fold'];arm=row['arm'];configs=ORD if arm=='ordinal' else GATE;tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f)
        assert p.sha(row['context'])==row['context_sha256'];ctx=np.load(row['context']);np.testing.assert_array_equal(ctx['train'],tr);np.testing.assert_array_equal(ctx['test'],te)
        splits=list(StratifiedKFold(3,shuffle=True,random_state=20261105).split(tr,np.digitize(d['y'][tr],[6,14])));A=np.full((len(tr),len(configs)),np.nan)
        for item in row['inner']+[{'path':row['path'],'sha256':row['sha256'],'final':True}]:
            final=item.get('final',False)
            if final:ti=tr;vi=te
            else:
                it,iv=splits[item['inner']];ti=tr[it];vi=tr[iv];np.testing.assert_array_equal(ti,item['train']);np.testing.assert_array_equal(vi,item['test'])
            assert not set(d['ids'][ti])&set(d['ids'][vi]) and not set(d['ids'][ti])&set(d['ids'][te])
            assert p.sha(item['path'])==item['sha256'];m=joblib.load(item['path']);assert m['ntrain']==len(ti)
            np.testing.assert_allclose(m['scaler'].mean_,d['X'][ti].mean(0),rtol=0,atol=1e-12);np.testing.assert_allclose(m['scaler'].var_,d['X'][ti].var(0),rtol=0,atol=1e-10)
            expected=configs[row['selected']] if final else configs[item['candidate']];assert m['config']==expected
            if arm=='gate':assert m['group_counts']==[int(sum(d['y'][ti]<6)),int(sum(d['y'][ti]>=6))]
            pr=replay_prediction(m,d['X'][vi]);delta=float(max(abs(pr-predict(m,d['X'][vi]))));assert delta<=1e-10;maxdiff=max(maxdiff,delta)
            if final:np.testing.assert_allclose(pr,ctx['pred'],rtol=0,atol=1e-10);pred[arm][te]=pr
            else:A[iv,item['candidate']]=pr
            count+=1
        np.testing.assert_allclose(A,ctx['A'],rtol=0,atol=1e-10);errors=abs(A-d['y'][tr,None]).mean(0);assert int(np.argmin(errors))==row['selected'];np.testing.assert_allclose(errors,row['inner_mae'],rtol=0,atol=1e-10)
    assert count==190 and all(np.isfinite(x).all() for x in pred.values());summary={};contrasts={};goals={}
    for group,mask in [('all',np.ones(125,bool)),('age65plus',d['age']>=65)]:
        summary[group]={a:metrics(d['y'][mask],pr[mask]) for a,pr in pred.items()};contrasts[group]={f'{a}_minus_{b}':paired(d['y'][mask],pr[mask],old[b][mask]) for a,pr in pred.items() for b in ['A','B','C']}
        goals[group]={a:{'mae_pass':summary[group][a]['mae']<=1,'within2_pass':summary[group][a]['within2']>=.8,'relative_improvement_pass':contrasts[group][a+'_minus_A']['relative_improvement']>=.2,'paired_CI_pass':contrasts[group][a+'_minus_A']['ci95'][1]<0} for a in pred}
    p.save(H/'round3-results.json',{'status':'exploratory_reused_development_folds','units':'JH-FRAT_points','summary':summary,'paired':contrasts,'goals':goals,'goal_passing_models':[a for a in pred if all(all(goals[g][a].values()) for g in goals)]})
    np.savez_compressed(B/'round3-verified-predictions.npz',ids=d['ids'],y=d['y'],participant_age=d['age'],fold=d['fold'],**pred)
    p.save(H/'round3-verification.json',{'status':'passed','internal_models':180,'final_models':10,'person_overlap':0,'train_only_normalization_and_inner_selection':True,'all_linear_logistic_predictions_independently_recomputed':True,'max_prediction_difference':maxdiff,'results_sha256':p.sha(H/'round3-results.json')})
    print(json.dumps({g:{a:round(v['mae'],4) for a,v in z.items()} for g,z in summary.items()},indent=2),flush=True)

if __name__=='__main__':{'freeze':freeze,'train':train,'review':review}[sys.argv[1]]()
