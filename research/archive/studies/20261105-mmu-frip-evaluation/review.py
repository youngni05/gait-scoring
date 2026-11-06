"""All saved-fit replay, selection/normalization audit and independent held-out metrics."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,csv,hashlib,sys,warnings
from pathlib import Path
import numpy as np,joblib
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import prepare as p
import learning as l
warnings.filterwarnings('ignore',message='X does not have valid feature names')
H=p.H;B=p.B
def predict(m,d,idx,arm,double=False):
    if arm in ['A','B','age']:
        X=d['age'][idx,None] if arm=='age' else d['X'][idx,:1] if arm=='A' else d['X'][idx]
        z=(X-m['scaler'].mean_)/m['scaler'].scale_;return np.maximum(0,m['model'].predict(z)*m['ystd']+m['ymean'])
    if arm=='C':return l.predict_tcn(m,d['sequence'][idx],d['X'][idx,0],double)
    return l.predict_random(m,d['sequence'][idx],d['X'][idx,0])
def norm_check(m,d,tr,arm):
    if arm in ['A','B','age']:
        X=d['age'][tr,None] if arm=='age' else d['X'][tr,:1] if arm=='A' else d['X'][tr]
        np.testing.assert_allclose(m['scaler'].mean_,X.mean(0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(m['scaler'].var_,X.var(0),rtol=0,atol=1e-10)
        np.testing.assert_allclose([m['ymean'],m['ystd']],[d['y'][tr].mean(),max(d['y'][tr].std(),1e-6)],rtol=0,atol=1e-12)
    else:
        expected=l.normalize_sequence(d['sequence'][tr],d['X'][tr,0])
        for k in expected:np.testing.assert_array_equal(m['norm'][k],expected[k])
    assert m['ntrain']==len(tr)
def metrics(y,pr):
    err=pr-y;rng=np.random.default_rng(20261105);boot=abs(err)[rng.integers(0,len(y),(5000,len(y)))].mean(1);true=y>=6;positive=pr>=6
    tp=int(sum(true&positive));tn=int(sum(~true&~positive));fp=int(sum(~true&positive));fn=int(sum(true&~positive));rec=tp/max(1,tp+fn);spec=tn/max(1,tn+fp)
    return {'n':len(y),'mae':float(abs(err).mean()),'mae_ci95':np.quantile(boot,[.025,.975]).tolist(),'rmse':float(np.sqrt(np.mean(err**2))),'bias':float(err.mean()),'r2':float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),
      'spearman':float(np.corrcoef(rankdata(y),rankdata(pr))[0,1]) if pr.std()>0 else None,'within1':float(np.mean(abs(err)<=1)),'within2':float(np.mean(abs(err)<=2)),'within3':float(np.mean(abs(err)<=3)),'error_over3':int(sum(abs(err)>3)),
      'risk_auc':float(roc_auc_score(true,pr)) if len(set(true))==2 else None,'risk_balanced_accuracy':(rec+spec)/2,'risk_recall':rec,'risk_specificity':spec,'risk_confusion':{'TP':tp,'TN':tn,'FP':fp,'FN':fn},
      'bands':[{'low':lo,'high':hi,'n':int(sum((y>=lo)&(y<hi))),'mae':float(abs(err[(y>=lo)&(y<hi)]).mean()) if np.any((y>=lo)&(y<hi)) else None,'bias':float(err[(y>=lo)&(y<hi)].mean()) if np.any((y>=lo)&(y<hi)) else None} for lo,hi in [(0,6),(6,14),(14,999)]]}
def paired(y,a,b):
    diff=abs(a-y)-abs(b-y);rng=np.random.default_rng(20261105);boot=diff[rng.integers(0,len(y),(5000,len(y)))].mean(1)
    return {'delta_mae':float(diff.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'relative_improvement':float(1-abs(a-y).mean()/abs(b-y).mean())}
def main():
    prot=json.loads((H/'model-protocol.json').read_text());rec=json.loads((H/'fit-receipt.json').read_text());d=np.load(B/'input.npz')
    assert prot['code_sha256']==p.sha(H/'run.py')==rec['code_sha256'] and prot['library_sha256']==p.sha(H/'learning.py')==rec['library_sha256'] and prot['input_sha256']==p.sha(B/'input.npz');assert len(rec['folds'])==5
    pred={a:np.full(125,np.nan) for a in ['A','B','C','R','D','age','median','mean','C_shuffled']};maxdiff=0;max64=0;internal=0;finals=0;selections=[]
    for fold,r in enumerate(rec['folds']):
      tr=np.array(r['train']);te=np.array(r['test']);assert np.array_equal(tr,np.flatnonzero(d['fold']!=fold)) and np.array_equal(te,np.flatnonzero(d['fold']==fold));assert not set(d['ids'][tr])&set(d['ids'][te])
      assert p.sha(r['context'])==r['context_sha256'];ctx=np.load(r['context']);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261105).split(tr,np.digitize(d['y'][tr],[6,14])));mat={a:np.full((len(tr),len(configs)),np.nan) for a,configs in [('A',l.TAB),('B',l.TAB),('C',l.TCN),('R',l.RANDOM)]};pos={int(x):i for i,x in enumerate(tr)};epoch_records={}
      for item in r['inner']:
        assert p.sha(item['file'])==item['sha256'];m=joblib.load(item['file']);ti=np.array(item['train']);vi=np.array(item['test']);it,iv=splits[item['inner']];np.testing.assert_array_equal(ti,tr[it]);np.testing.assert_array_equal(vi,tr[iv]);assert not set(d['ids'][ti])&set(d['ids'][vi]) and not set(d['ids'][ti])&set(d['ids'][te]);norm_check(m,d,ti,item['arm'])
        pp=predict(m,d,vi,item['arm']);mat[item['arm']][iv,item['candidate']]=pp
        if item['arm']=='C':
            dp=predict(m,d,vi,'C',True);max64=max(max64,float(max(abs(dp-pp))));assert max(abs(dp-pp))<=2e-5
            best=float('inf');epoch=0
            for ep,loss in enumerate(m['history'],1):
                if loss<best-1e-6:best=loss;epoch=ep
            assert m['epochs']==epoch;epoch_records[(item['candidate'],item['inner'])]=epoch
        internal+=1
      for j,arm in enumerate(['A','B','C','R']):
        errors=abs(mat[arm]-d['y'][tr,None]).mean(0);k=int(np.argmin(errors));assert k==r['selected'][arm]['candidate'];np.testing.assert_allclose(errors,r['selected'][arm]['all_candidate_inner_mae'],rtol=0,atol=1e-10);np.testing.assert_allclose(mat[arm][:,k],ctx['innerOOF'][:,j],rtol=0,atol=1e-10)
      for item in r['final']:
        assert p.sha(item['file'])==item['sha256'];m=joblib.load(item['file']);arm=item['arm'];norm_check(m,d,tr,arm);pp=predict(m,d,te,arm);delta=float(max(abs(pp-ctx[arm])));maxdiff=max(maxdiff,delta);assert delta<=1e-10;pred[arm][te]=pp;finals+=1
        if arm=='C':
          k=r['selected']['C']['candidate'];assert m['epochs']==int(np.median([epoch_records[(k,i)] for i in range(3)]));double=predict(m,d,te,'C',True);max64=max(max64,float(max(abs(double-pp))));assert max(abs(double-pp))<=2e-5
          order=np.random.default_rng(20261105).permutation(60);sh=l.predict_tcn(m,d['sequence'][te][:,:,order],d['X'][te,0]);np.testing.assert_allclose(sh,ctx['C_shuffled'],rtol=0,atol=1e-10);pred['C_shuffled'][te]=sh
      A=ctx['innerOOF'];center=A.mean(0);scale=np.maximum(A.std(0),1e-6);ym=d['y'][tr].mean();coef=nnls(np.vstack([(A-center)/scale,np.eye(4)]),np.r_[d['y'][tr]-ym,np.zeros(4)])[0]
      assert p.sha(r['blend'])==r['blend_sha256'];blend=joblib.load(r['blend']);np.testing.assert_allclose(coef,blend['coef'],rtol=0,atol=1e-12);pp=np.maximum(0,(ctx['outerbases']-center)/scale@coef+ym);np.testing.assert_allclose(pp,ctx['D'],rtol=0,atol=1e-10);pred['D'][te]=pp
      for arm,value in [('median',np.median(d['y'][tr])),('mean',d['y'][tr].mean())]:pred[arm][te]=value;np.testing.assert_array_equal(ctx[arm],pred[arm][te])
      selections.append({'fold':fold,'selected':r['selected'],'blend_coefficients':coef.tolist()});print('REPLAYED_FOLD',fold,flush=True)
    assert internal==375 and finals==25 and all(np.isfinite(a).all() for a in pred.values())
    summaries={};contrasts={};goals={}
    for group,mask in [('all',np.ones(125,bool)),('age65plus',d['age']>=65)]:
      y=d['y'][mask];summaries[group]={a:metrics(y,pp[mask]) for a,pp in pred.items()};contrasts[group]={f'{a}_minus_{b}':paired(y,pred[a][mask],pred[b][mask]) for a,b in [('B','A'),('C','A'),('R','A'),('D','A'),('D','B'),('D','C'),('C_shuffled','C')]}
      goals[group]={a:dict(mae_pass=summaries[group][a]['mae']<=1,within2_pass=summaries[group][a]['within2']>=.8,relative_improvement_pass=contrasts[group][a+'_minus_A']['relative_improvement']>=.2,paired_CI_pass=contrasts[group][a+'_minus_A']['ci95'][1]<0) for a in ['B','C','R','D']}
    goal_pass=[a for a in ['B','C','R','D'] if all(all(goals[g][a].values()) for g in goals)]
    rows=[]
    for i,pid in enumerate(d['ids']):rows.append({'id':pid,'age':float(d['age'][i]),'fold':int(d['fold'][i]),'score':float(d['y'][i]),**{a:float(pp[i]) for a,pp in pred.items()}})
    with (H/'person-predictions.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p.save(H/'results.json',{'units':'JH-FRATpoints,notseconds','people':125,'older65':56,'conditions':summaries,'paired':contrasts,'goals':goals,'goal_passing_models':goal_pass,'selections':selections,'status':'nested_development_evaluation_notclinicalvalidation'})
    np.savez_compressed(B/'verified-predictions.npz',ids=d['ids'],y=d['y'],participant_age=d['age'],fold=d['fold'],**pred)
    p.save(H/'verification.json',{'status':'passed','internal_saved_models_replayed':internal,'final_saved_models_replayed':finals,'blends_rebuilt':5,'person_overlap':0,'train_only_normalization_and_epoch_selection_verified':True,'max_prediction_difference':maxdiff,'max_TCN_CPU64_difference':max64,'results_sha256':p.sha(H/'results.json'),'person_predictions_sha256':p.sha(H/'person-predictions.csv')})
    print(json.dumps({'summary':{g:{a:{k:v for k,v in z.items() if k in ['mae','mae_ci95','within2','risk_auc','risk_balanced_accuracy']} for a,z in aa.items()} for g,aa in summaries.items()},'goal_pass':goal_pass,'max_CPU64':max64},indent=2),flush=True)
if __name__=='__main__':main()
