"""Follow-up attribution: same CNN, features and loss, remove person-specific motion."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys
import numpy as np,joblib
import prepare as p
import learning as l
import run as runner
from review import metrics,paired
H=p.H;B=p.B

def freeze():
    assert not(H/'ablation-protocol.json').exists();r=p.read(H/'results.json');v=p.read(H/'verification.json');assert v['status']=='passed' and p.sha(H/'results.json')==v['results_sha256']
    p.save(H/'ablation-protocol.json',{'code_sha256':p.sha(__file__),'primary_results_sha256':p.sha(H/'results.json'),'learning_sha256':p.sha(H/'learning.py'),'original_protocol_sha256':p.sha(H/'protocol.json'),
      'reason':'D improves all-person MAE vs speed but shuffling time changes only0.006sec. Separate temporal/person-specific pose contribution from optimizer/anchor/feature effects.',
      'status':'Diagnosticfollowupafterprimaryresultseen;samepreviouslyreuseddevelopmentfolds,notexternalconfirmation.',
      'arms':{'C0':'Replaceallpersonmotionbyconstantzero34x91;sametrueone-speedinputasC','D0':'Replaceallpersonmotionbyconstantzero34x91;sametrue64featuresasD'},
      'method':'ExactsameTCNarchitecture,noise,optimizer,4width/targetconfigs,seeds,maxepochs100/patience15,inner3choiceandfinalmedianepoch.Onlymotionsignalremoved. Constantzeroisintentionalcounterfactualcontrol,notanimputedphysicalmeasurement.',
      'counts':{'internal':120,'final':10},'evaluation':'Originalrepeat0same5folds,all297andolder34,severe9. CompareDminusD0,CminusC0andD0minusA/B. No test-based sample exclusions or target changes.',
      'verification':'All130savedfitsSHA,reload,train-onlynorm/targetstats,innerselection/epochs,CPU64<=2e-5,personsplit,independentmetrics.',
      'interpretation':'IfD0issimilartoD,do notattributeDgainspecificallytolearnedmotion. Followupnotindependentproof. No ad hoc outer-tuning aftercontrol.'})

def train():
    protocol=p.read(H/'ablation-protocol.json');assert protocol['code_sha256']==p.sha(__file__);_,d,folds=runner.check();S=np.zeros_like(d['sequence']);folder=B/'ablation-models';folder.mkdir(exist_ok=True);records=[]
    for fold in folds:
      f=fold['fold'];tr=np.array(fold['train']);te=np.array(fold['test'])
      for arm in ['C0','D0']:
        X=d['X'][:,:1] if arm=='C0' else d['X'];A=np.full((len(tr),4),np.nan);positions={int(v):i for i,v in enumerate(tr)};inner=[];epochs={k:[] for k in range(4)}
        for k,cfg in enumerate(l.TCN):
          for j,part in enumerate(fold['inner']):
            ti=np.array(part['train']);vi=np.array(part['validation']);m=l.fit_tcn(S[ti],X[ti],d['y'][ti],cfg,20261105+f*100+k,validation=(S[vi],X[vi],d['y'][vi]));path=folder/f'f{f}-{arm}-k{k}-i{j}.joblib';joblib.dump(m,path,compress=3)
            A[[positions[int(v)] for v in vi],k]=l.predict_tcn(m,S[vi],X[vi]);epochs[k].append(m['epochs']);inner.append({'candidate':k,'inner':j,'train':ti.tolist(),'test':vi.tolist(),'path':str(path),'sha256':p.sha(path)})
        error=abs(A-d['y'][tr,None]).mean(0);k=int(np.argmin(error));m=l.fit_tcn(S[tr],X[tr],d['y'][tr],l.TCN[k],20261105+f*100+k,epochs=max(1,int(np.median(epochs[k]))));path=folder/f'f{f}-{arm}-final.joblib';joblib.dump(m,path,compress=3);ctx=folder/f'f{f}-{arm}-context.npz';np.savez_compressed(ctx,train=tr,test=te,A=A,pred=l.predict_tcn(m,S[te],X[te]))
        records.append({'fold':f,'arm':arm,'selected':k,'inner_mae':error.tolist(),'inner':inner,'path':str(path),'sha256':p.sha(path),'context':str(ctx),'context_sha256':p.sha(ctx)})
        p.save(H/'ablation-receipt.json',{'records':records,'protocol_sha256':p.sha(H/'ablation-protocol.json')});print('ABLATION_FIT',f,arm,flush=True)

def review():
    protocol=p.read(H/'ablation-protocol.json');assert protocol['code_sha256']==p.sha(__file__) and protocol['primary_results_sha256']==p.sha(H/'results.json') and protocol['learning_sha256']==p.sha(H/'learning.py')
    _,d,folds=runner.check();S=np.zeros_like(d['sequence']);base=np.load(B/'verified-predictions.npz');receipt=p.read(H/'ablation-receipt.json');assert receipt['protocol_sha256']==p.sha(H/'ablation-protocol.json') and len(receipt['records'])==10
    pred={a:np.full(297,np.nan) for a in ['C0','D0']};count=0;max64=0.;selections=[]
    for row in receipt['records']:
        f=row['fold'];fold=folds[f];arm=row['arm'];X=d['X'][:,:1] if arm=='C0' else d['X'];tr=np.array(fold['train']);te=np.array(fold['test']);assert p.sha(row['context'])==row['context_sha256'];ctx=np.load(row['context']);np.testing.assert_array_equal(tr,ctx['train']);np.testing.assert_array_equal(te,ctx['test'])
        A=np.full((len(tr),4),np.nan);positions={int(v):i for i,v in enumerate(tr)};epochs={}
        for item in row['inner']+[{'path':row['path'],'sha256':row['sha256'],'final':True}]:
            final=item.get('final',False)
            if final:ti=tr;vi=te;k=row['selected']
            else:
                part=fold['inner'][item['inner']];ti=np.array(part['train']);vi=np.array(part['validation']);np.testing.assert_array_equal(ti,item['train']);np.testing.assert_array_equal(vi,item['test']);k=item['candidate']
            assert not set(d['ids'][ti])&set(d['ids'][vi]) and not set(d['ids'][ti])&set(d['ids'][te]);assert p.sha(item['path'])==item['sha256'];m=joblib.load(item['path']);assert m['config']==l.TCN[k] and m['ntrain']==len(ti)
            expected=l.norm(S[ti],X[ti])
            for key in ['seqmean','seqstd','median']:np.testing.assert_array_equal(m['norm'][key],expected[key])
            np.testing.assert_allclose(m['norm']['scaler'].mean_,expected['scaler'].mean_,rtol=0,atol=1e-12);np.testing.assert_allclose(m['norm']['scaler'].var_,expected['scaler'].var_,rtol=0,atol=1e-10)
            yy=d['y'][ti]-l.anchor(X[ti],m['config']);np.testing.assert_allclose([m['ymean'],m['ystd']],[yy.mean(),max(yy.std(),1e-6)],rtol=0,atol=1e-12)
            pr=l.predict_tcn(m,S[vi],X[vi]);double=l.predict_tcn(m,S[vi],X[vi],True);delta=float(max(abs(double-pr)));assert delta<=2e-5;max64=max(max64,delta)
            if final:
                assert m['epochs']==max(1,int(np.median([epochs[(k,j)] for j in range(3)])));np.testing.assert_allclose(pr,ctx['pred'],rtol=0,atol=1e-10);pred[arm][te]=pr
            else:
                best=float('inf');ep=0
                for j,error in enumerate(m['history'],1):
                    if error<best-1e-6:best=error;ep=j
                assert ep==m['epochs'];epochs[(k,item['inner'])]=ep;A[[positions[int(v)] for v in vi],k]=pr
            count+=1
        assert np.isfinite(A).all();np.testing.assert_allclose(A,ctx['A'],rtol=0,atol=1e-10);error=abs(A-d['y'][tr,None]).mean(0);assert int(np.argmin(error))==row['selected'];np.testing.assert_allclose(error,row['inner_mae'],rtol=0,atol=1e-10);selections.append({'fold':f,'arm':arm,'candidate':row['selected']});print('ABLATION_REPLAY',f,arm,flush=True)
    assert count==130 and all(np.isfinite(a).all() for a in pred.values());summary={};contrasts={}
    for group,mask in [('all',np.ones(297,bool)),('age65plus',d['age']>=65),('TUG20plus',d['y']>=20)]:
        y=d['y'][mask];summary[group]={a:metrics(y,v[mask]) for a,v in pred.items()};contrasts[group]={f'{a}_minus_{b}':paired(y,x[mask],z[mask]) for a,b,x,z in [('C','C0',base['C'],pred['C0']),('D','D0',base['D'],pred['D0']),('C0','A',pred['C0'],base['A']),('D0','A',pred['D0'],base['A']),('D0','B',pred['D0'],base['B'])]}
    p.save(H/'ablation-results.json',{'status':'diagnostic_reused_development_after_primary','summary':summary,'paired':contrasts,'selections':selections,'interpretation':'Constantzeroinputisacounterfactualcontrol. DminusD0testsaddedperson-specificposeunderthisarchitecture/training;notstrictindependentprooforparameter-countbenchmark.'})
    np.savez_compressed(B/'ablation-verified-predictions.npz',ids=d['ids'],y=d['y'],participant_age=d['age'],**pred)
    p.save(H/'ablation-verification.json',{'status':'passed','internal_models_replayed':120,'final_models_replayed':10,'person_overlap':0,'train_only_normalization_and_inner_selection':True,'max_TCN_CPU64_difference':max64,'frozen_tolerance':2e-5,'results_sha256':p.sha(H/'ablation-results.json')})
    print({g:{a:round(v['MAE'],4) for a,v in z.items()} for g,z in summary.items()},flush=True)

if __name__=='__main__':{'freeze':freeze,'train':train,'review':review}[sys.argv[1]]()
