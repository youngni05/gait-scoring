"""Replay every exploratory 3D fit; compare only common held-out people."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,warnings
import numpy as np,joblib
from sklearn.model_selection import StratifiedKFold
import prepare as p
import learning as l
import round2_3d as r
from review import metrics,paired
warnings.filterwarnings('ignore',message='X does not have valid feature names')
H=p.H;B=p.B

def norm_check(m,d,base,tr,arm):
    if arm=='C_3D':
        expected=l.normalize_sequence(d['sequence3D'][tr],base['X'][tr,0])
        for key in expected:np.testing.assert_array_equal(m['norm'][key],expected[key])
    else:
        X=d['Xprovider'][tr] if arm=='S_provider' else d['Xcombined'][tr]
        np.testing.assert_allclose(m['scaler'].mean_,X.mean(0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(m['scaler'].var_,X.var(0),rtol=0,atol=1e-10)
    np.testing.assert_allclose([m['ymean'],m['ystd']],[d['y'][tr].mean(),max(d['y'][tr].std(),1e-6)],rtol=0,atol=1e-12)
    assert m['ntrain']==len(tr)

def predict(m,d,base,idx,arm,double=False):
    if arm=='C_3D':return r.pred3(m,d['sequence3D'][idx],base['X'][idx,0],double)
    X=d['Xprovider'][idx] if arm=='S_provider' else d['Xcombined'][idx]
    z=(X-m['scaler'].mean_)/m['scaler'].scale_
    return np.maximum(0,m['model'].predict(z)*m['ystd']+m['ymean'])

def main():
    prot=json.loads((H/'round2-protocol.json').read_text());receipt=json.loads((H/'round2-receipt.json').read_text());audit=json.loads((H/'round2-input-audit.json').read_text())
    assert prot['code_sha256']==p.sha(H/'round2_3d.py')==receipt['code_sha256']
    assert prot['round1_results_sha256']==p.sha(H/'results.json')
    assert receipt['input_sha256']==audit['input_sha256']==p.sha(B/'round2-input.npz')
    d=np.load(B/'round2-input.npz');base=np.load(B/'input.npz');old=np.load(B/'verified-predictions.npz')
    for key in ['ids','y','fold']:np.testing.assert_array_equal(d[key],base[key]);np.testing.assert_array_equal(d[key],old[key])
    assert len(receipt['records'])==15
    pred={a:np.full(125,np.nan) for a in ['B_2D3D','C_3D','S_provider']};internal=0;finals=0;maxdiff=0.;max64=0.;counts=[]
    for row in receipt['records']:
        f=row['fold'];arm=row['arm'];mask=np.ones(125,bool) if arm=='S_provider' else d['eligible3D']
        tr=np.flatnonzero((d['fold']!=f)&mask);te=np.flatnonzero((d['fold']==f)&mask)
        assert not set(d['ids'][tr])&set(d['ids'][te])
        assert p.sha(row['context'])==row['context_sha256'];ctx=np.load(row['context'])
        np.testing.assert_array_equal(tr,ctx['train']);np.testing.assert_array_equal(te,ctx['test'])
        configs=l.TCN if arm=='C_3D' else l.TAB;A=np.full((len(tr),len(configs)),np.nan)
        splits=list(StratifiedKFold(3,shuffle=True,random_state=20261105).split(tr,np.digitize(d['y'][tr],[6,14])))
        epochs={}
        assert len(row['inner'])==len(configs)*3
        for item in row['inner']:
            assert p.sha(item['path'])==item['sha256'];m=joblib.load(item['path']);it,iv=splits[item['inner']]
            ti=np.array(item['train']);vi=np.array(item['test']);np.testing.assert_array_equal(ti,tr[it]);np.testing.assert_array_equal(vi,tr[iv])
            assert not set(d['ids'][ti])&set(d['ids'][vi]) and not set(d['ids'][ti])&set(d['ids'][te])
            norm_check(m,d,base,ti,arm);pr=predict(m,d,base,vi,arm);A[iv,item['candidate']]=pr;internal+=1
            if arm=='C_3D':
                dp=predict(m,d,base,vi,arm,True);delta=float(max(abs(dp-pr)));max64=max(max64,delta);assert delta<=2e-5
                best=float('inf');ep=0
                for j,loss in enumerate(m['history'],1):
                    if loss<best-1e-6:best=loss;ep=j
                assert ep==m['epochs'];epochs[(item['candidate'],item['inner'])]=ep
        assert np.isfinite(A).all();np.testing.assert_allclose(A,ctx['A'],rtol=0,atol=1e-10)
        errors=abs(A-d['y'][tr,None]).mean(0);k=int(np.argmin(errors));assert k==row['selected'];np.testing.assert_allclose(errors,row['inner_mae'],rtol=0,atol=1e-10)
        assert p.sha(row['path'])==row['sha256'];m=joblib.load(row['path']);norm_check(m,d,base,tr,arm)
        assert m['config']==configs[k]
        if arm=='C_3D':assert m['epochs']==max(1,int(np.median([epochs[(k,j)] for j in range(3)])))
        pr=predict(m,d,base,te,arm);delta=float(max(abs(pr-ctx['pred'])));maxdiff=max(maxdiff,delta);assert delta<=1e-10
        if arm=='C_3D':
            dp=predict(m,d,base,te,arm,True);delta=float(max(abs(dp-pr)));max64=max(max64,delta);assert delta<=2e-5
        pred[arm][te]=pr;finals+=1;counts.append({'fold':f,'arm':arm,'train':len(tr),'test':len(te),'candidate':k});print('ROUND2_REPLAY',f,arm,flush=True)
    assert internal==330 and finals==15
    summaries={};contrasts={}
    for group,age in [('all',np.ones(125,bool)),('age65plus',d['age']>=65)]:
        common=age&d['eligible3D'];allpred={a:old[a] for a in ['A','B','C']};allpred.update({a:pred[a] for a in ['B_2D3D','C_3D']})
        assert all(np.isfinite(pp[common]).all() for pp in allpred.values())
        summaries[group]={'same_3D_eligible_people':{a:metrics(d['y'][common],pp[common]) for a,pp in allpred.items()},'all125_provider_diagnostic':metrics(d['y'][age],pred['S_provider'][age])}
        contrasts[group]={f'{a}_minus_{b}':paired(d['y'][common],allpred[a][common],allpred[b][common]) for a,b in [('B_2D3D','A'),('B_2D3D','B'),('C_3D','A'),('C_3D','C')]}
        contrasts[group]['S_provider_minus_A_all125']=paired(d['y'][age],pred['S_provider'][age],old['A'][age])
    p.save(H/'round2-results.json',{'status':'exploratory_reused_development_folds_not_independent_confirmation','units':'JH-FRAT_points','matched3D':audit['matched3D_people'],'matched3D_older65':audit['matched3D_older65'],'unmeasurable_3D':audit['failures'],'summary':summaries,'paired':contrasts,'fold_counts':counts,'limitations':'B/C3Dtrainedon3Deligibleonly;round1A/B/Ctrainedontheoriginal125,matchedtestpeopleonly. Thiscannotisolatethecausaleffectof3Dfromtrainingcohortchange. ProviderfeaturesusefullTUG,notshortwalking.'})
    np.savez_compressed(B/'round2-verified-predictions.npz',ids=d['ids'],y=d['y'],participant_age=d['age'],fold=d['fold'],eligible3D=d['eligible3D'],**pred)
    p.save(H/'round2-verification.json',{'status':'passed','internal_saved_models_replayed':internal,'final_saved_models_replayed':finals,'person_overlap':0,'inner_selection_and_train_only_normalization':True,'max_prediction_difference':maxdiff,'max_TCN_CPU64_difference':max64,'frozen_CPU64_tolerance':2e-5,'results_sha256':p.sha(H/'round2-results.json')})
    compact={g:{k:({a:round(m['mae'],4) for a,m in v.items()} if k=='same_3D_eligible_people' else round(v['mae'],4)) for k,v in s.items()} for g,s in summaries.items()}
    print(json.dumps({'matched3D':audit['matched3D_people'],'summary':compact},indent=2),flush=True)

if __name__=='__main__':main()
