"""Exhaustive saved-fit replay and independently rebuilt person-level TUG metrics."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,csv,warnings
import numpy as np,joblib
from scipy.optimize import nnls
from scipy.stats import rankdata
import prepare as p
import learning as l
import run as runner
warnings.filterwarnings('ignore',message='X does not have valid feature names')
H=p.H;B=p.B

def metrics(y,pred):
    err=pred-y;rng=np.random.default_rng(20261105);boots=abs(err)[rng.integers(0,len(y),(10000,len(y)))].mean(1)
    return {'n':len(y),'MAE':float(abs(err).mean()),'MAE_CI95':np.quantile(boots,[.025,.975]).tolist(),'RMSE':float(np.sqrt(np.mean(err**2))),'bias':float(err.mean()),'median_AE':float(np.median(abs(err))),'p95_AE':float(np.quantile(abs(err),.95)),'within1':float(np.mean(abs(err)<=1)),'within2':float(np.mean(abs(err)<=2)),'error_over3':int(sum(abs(err)>3)),'spearman':float(np.corrcoef(rankdata(y),rankdata(pred))[0,1]) if pred.std()>0 else None}
def paired(y,a,b):
    delta=abs(a-y)-abs(b-y);rng=np.random.default_rng(20261105);boot=delta[rng.integers(0,len(y),(10000,len(y)))].mean(1)
    return {'delta_MAE':float(delta.mean()),'CI95':np.quantile(boot,[.025,.975]).tolist(),'relative_improvement':float(1-abs(a-y).mean()/abs(b-y).mean())}

def norm_check(m,d,tr,arm):
    X=d['X'][tr,:1] if arm=='C' else d['X'][tr];S=d['sequence'][tr];expected=l.norm(S,X)
    for key in ['seqmean','seqstd','median']:np.testing.assert_array_equal(m['norm'][key],expected[key])
    np.testing.assert_allclose(m['norm']['scaler'].mean_,expected['scaler'].mean_,rtol=0,atol=1e-12);np.testing.assert_allclose(m['norm']['scaler'].var_,expected['scaler'].var_,rtol=0,atol=1e-10)
    target=d['y'][tr] if arm=='R' else d['y'][tr]-l.anchor(X,m['config'])
    np.testing.assert_allclose([m['ymean'],m['ystd']],[target.mean(),max(target.std(),1e-6)],rtol=0,atol=1e-12);assert m['ntrain']==len(tr)
    if arm=='R':
        z=(S-m['norm']['seqmean'])/m['norm']['seqstd'];xx=np.c_[l.tab(m['norm'],X),l.random_features(z,m['kernels'])]
        np.testing.assert_allclose(m['scaler'].mean_,xx.mean(0),rtol=0,atol=1e-10);np.testing.assert_allclose(m['scaler'].var_,xx.var(0),rtol=0,atol=1e-9)

def baseline_norm_check(m,X):
    imputer=m.steps[0][1];scaler=m.steps[1][1];med=np.nanmedian(X,axis=0);med=np.nan_to_num(med,nan=0.)
    np.testing.assert_allclose(imputer.statistics_,med,rtol=0,atol=1e-12)
    filled=np.where(np.isnan(X),med,X);np.testing.assert_allclose(scaler.mean_,filled.mean(0),rtol=0,atol=1e-12);np.testing.assert_allclose(scaler.var_,filled.var(0),rtol=0,atol=1e-10)

def main():
    prot,d,folds=runner.check();receipt=p.read(H/'fit-receipt.json');assert len(receipt['records'])==5 and receipt['protocol_sha256']==p.sha(H/'protocol.json')
    assert len(set(d['ids']))==297 and np.isfinite(d['y']).all() and np.isfinite(d['sequence']).all()
    assert receipt['input_sha256']==p.sha(B/'input.npz')
    pred={arm:np.full(297,np.nan) for arm in ['A','A2','B','C','D','R','E','C_shuffled','D_shuffled']};counts={'internal':0,'final':0,'blends':0,'old_baselines':0};maxdiff=0.;max64=0.;selected=[]
    oldrows=list(csv.DictReader((p.OLD/'person-predictions.csv').open(encoding='utf8')));assert [r['id'] for r in oldrows]==d['ids'].tolist()
    for row,fold in zip(receipt['records'],folds):
        f=fold['fold'];assert row['fold']==f;tr=np.array(fold['train']);te=np.array(fold['test']);np.testing.assert_array_equal(tr,row['train']);np.testing.assert_array_equal(te,row['test'])
        assert not set(d['ids'][tr])&set(d['ids'][te]);assert p.sha(row['context'])==row['context_sha256'];ctx=np.load(row['context']);np.testing.assert_array_equal(tr,ctx['train']);np.testing.assert_array_equal(te,ctx['test'])
        matrices={a:np.full((len(tr),len(l.RANDOM if a=='R' else l.TCN)),np.nan) for a in ['C','D','R']};positions={int(value):i for i,value in enumerate(tr)};epochs={}
        assert len(row['inner'])==33
        for item in row['inner']:
            arm=item['arm'];ti=np.array(item['train']);vi=np.array(item['test']);expected=fold['inner'][item['inner']];np.testing.assert_array_equal(ti,expected['train']);np.testing.assert_array_equal(vi,expected['validation'])
            assert not set(d['ids'][ti])&set(d['ids'][vi]) and not set(d['ids'][ti])&set(d['ids'][te]);assert set(ti)|set(vi)==set(tr)
            assert p.sha(item['path'])==item['sha256'];m=joblib.load(item['path']);configs=l.RANDOM if arm=='R' else l.TCN;assert m['config']==configs[item['candidate']];norm_check(m,d,ti,arm)
            pr=runner.predict(m,d,vi,arm);matrices[arm][[positions[int(i)] for i in vi],item['candidate']]=pr;counts['internal']+=1
            if arm!='R':
                double=runner.predict(m,d,vi,arm,True);delta=float(max(abs(double-pr)));max64=max(max64,delta);assert delta<=2e-5
                best=float('inf');ep=0
                for j,error in enumerate(m['history'],1):
                    if error<best-1e-6:best=error;ep=j
                assert m['epochs']==ep;epochs[(arm,item['candidate'],item['inner'])]=ep
        for arm,A in matrices.items():
            assert np.isfinite(A).all();loss=abs(A-d['y'][tr,None]).mean(0);k=int(np.argmin(loss));assert k==row['selected'][arm]['candidate'];np.testing.assert_allclose(loss,row['selected'][arm]['inner_mae'],rtol=0,atol=1e-10);np.testing.assert_allclose(A[:,k],ctx['inner_'+arm],rtol=0,atol=1e-10)
        for item in row['final']:
            assert p.sha(item['path'])==item['sha256'];arm=item['arm'];m=joblib.load(item['path']);k=row['selected'][arm]['candidate'];assert m['config']==(l.RANDOM if arm=='R' else l.TCN)[k];norm_check(m,d,tr,arm)
            pr=runner.predict(m,d,te,arm);delta=float(max(abs(pr-ctx[arm])));maxdiff=max(maxdiff,delta);assert delta<=1e-10;pred[arm][te]=pr;counts['final']+=1
            if arm!='R':
                assert m['epochs']==max(1,int(np.median([epochs[(arm,k,j)] for j in range(3)])))
                double=runner.predict(m,d,te,arm,True);delta=float(max(abs(double-pr)));max64=max(max64,delta);assert delta<=2e-5
                order=np.random.default_rng(20261105).permutation(91);X=d['X'][te,:1] if arm=='C' else d['X'][te];pred[arm+'_shuffled'][te]=l.predict_tcn(m,d['sequence'][te][:,:,order],X)
        prior=runner.old_models(d,fold)
        for arm,view,cols in [('A','speed_only',[0]),('A2','speed_dynamics',list(range(4))),('B','speed_plus_landmarks',list(range(64)))]:
            pp=prior[arm]['test'];np.testing.assert_allclose(pp,ctx[arm],rtol=0,atol=1e-10);np.testing.assert_allclose(prior[arm]['inner'],ctx['inner_'+arm],rtol=0,atol=1e-10)
            np.testing.assert_allclose(pp,np.array([float(oldrows[i]['r0_'+view]) for i in te]),rtol=0,atol=1e-10);pred[arm][te]=pp;counts['old_baselines']+=1
            baseline_norm_check(joblib.load(prior[arm]['model']),d['X'][tr][:,cols])
        A=np.column_stack([ctx['inner_'+arm] for arm in ['B','C','D','R']]);center=A.mean(0);scale=np.maximum(A.std(0),1e-6);ym=d['y'][tr].mean();coef=nnls(np.vstack([(A-center)/scale,np.eye(4)]),np.r_[d['y'][tr]-ym,np.zeros(4)])[0]
        assert p.sha(row['blend'])==row['blend_sha256'];blend=joblib.load(row['blend']);np.testing.assert_allclose(coef,blend['coef'],rtol=0,atol=1e-12);np.testing.assert_array_equal(center,blend['center']);np.testing.assert_array_equal(scale,blend['scale']);assert ym==blend['ymean']
        pp=np.maximum(0,(np.column_stack([pred[arm][te] for arm in ['B','C','D','R']])-center)/scale@coef+ym);np.testing.assert_allclose(pp,ctx['E'],rtol=0,atol=1e-10);pred['E'][te]=pp;counts['blends']+=1
        selected.append({'fold':f,'candidates':row['selected'],'blend_coefficients':coef.tolist()});print('REPLAYED_FOLD',f,flush=True)
    assert counts=={'internal':165,'final':15,'blends':5,'old_baselines':15} and all(np.isfinite(pp).all() for pp in pred.values())
    groups={'all':np.ones(297,bool),'age65plus':d['age']>=65,'TUGunder12':d['y']<12,'TUG12to20':(d['y']>=12)&(d['y']<20),'TUG20plus':d['y']>=20,'CTL':d['diagnosis']=='CTL'}
    summaries={};contrasts={};goals={}
    for group,mask in groups.items():
        if not np.any(mask):continue
        y=d['y'][mask];summaries[group]={arm:metrics(y,pr[mask]) for arm,pr in pred.items()};contrasts[group]={f'{arm}_minus_{base}':paired(y,pred[arm][mask],pred[base][mask]) for arm in ['C','D','R','E'] for base in ['A','A2','B']}
        for arm in ['C','D']:contrasts[group][arm+'_shuffled_minus_'+arm]=paired(y,pred[arm+'_shuffled'][mask],pred[arm][mask])
        if group in ['all','age65plus']:
            goals[group]={arm:{'MAE_le1':summaries[group][arm]['MAE']<=1,'within2_ge80percent':summaries[group][arm]['within2']>=.8,'relative_reduction_ge20percent':contrasts[group][arm+'_minus_A']['relative_improvement']>=.2,'paired_A_CI_improvement':contrasts[group][arm+'_minus_A']['CI95'][1]<0,'paired_A2_CI_improvement':contrasts[group][arm+'_minus_A2']['CI95'][1]<0} for arm in ['C','D','R','E']}
    goalpass=[arm for arm in ['C','D','R','E'] if all(all(goals[g][arm].values()) for g in goals)]
    promising=[arm for arm in ['C','D','R','E'] if arm in goalpass or contrasts['all'][arm+'_minus_B']['CI95'][1]<0]
    p.save(H/'results.json',{'status':'reused_development_primary_repeat0_only','units':'TUG_seconds','people':297,'age65plus':34,'TUG20plus':9,'summary':summaries,'paired':contrasts,'goals':goals,'goal_passing_arms':goalpass,'promising_for_four_repeat_sensitivity':promising,'selections':selected,'uncertainty':'10000pairedpersonbootstrapconditionalonfixedOOF;notfullretrainingorindependentexternalvalidation;multiplecomparisonsnotadjusted'})
    np.savez_compressed(B/'verified-predictions.npz',ids=d['ids'],y=d['y'],participant_age=d['age'],**pred)
    rows=[]
    for i,pid in enumerate(d['ids']):rows.append({'id':str(pid),'true_TUG':float(d['y'][i]),'age':float(d['age'][i]),**{arm:float(pp[i]) for arm,pp in pred.items()}})
    with (H/'person-predictions.csv').open('w',newline='',encoding='utf8') as file:writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    p.save(H/'verification.json',{'status':'passed',**counts,'person_overlap':0,'same297_source_and_windows':True,'train_only_norm_target_and_epoch_selection':True,'max_saved_prediction_difference':maxdiff,'max_TCN_CPU64_difference':max64,'CPU64_tolerance':2e-5,'results_sha256':p.sha(H/'results.json'),'prediction_table_sha256':p.sha(H/'person-predictions.csv'),'input_sha256':p.sha(B/'input.npz'),'review_code_sha256':p.sha(__file__)})
    print(json.dumps({'MAE':{group:{a:round(m['MAE'],4) for a,m in values.items() if not a.endswith('shuffled')} for group,values in summaries.items()},'goalpass':goalpass,'promising':promising,'CPU64max':max64},indent=2),flush=True)

if __name__=='__main__':main()
