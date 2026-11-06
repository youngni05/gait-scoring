import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import numpy as np,joblib
import sensor_loop as s
import refine_ensemble as r
from verify_batch import metric,delta
HERE=s.HERE;HEAVY=s.HEAVY
def main():
    prot=json.loads((HERE/'refinement-protocol.json').read_text());assert s.sha(HERE/'refine_ensemble.py')==prot['code_sha256']
    assert s.sha(HERE/'sensor-verification.json')==prot['source_new_verification_sha256']
    d=np.load(s.SOURCE);y=d['y'];ids=d['ids'];records=json.loads((HERE/'refinement-receipt.json').read_text());assert len(records)==15
    newrecords=json.loads((HERE/'sensor-receipt.json').read_text());stored=np.load(HEAVY/'refinement/predictions.npz');pred={};oldpred={};count=0;maxdiff=0.
    for rec in records:
        tr=np.array(rec['train']);te=np.array(rec['test']);f=rec['fold'];view=rec['view'];assert not set(ids[tr])&set(ids[te]);assert np.array_equal(te,np.flatnonzero(d['fold']==f))
        src=json.loads(Path(rec['source_old_record']).read_text());op=Path(src['context']);assert s.sha(op)==rec['source_old_context_sha256']==src['context_sha256'];oc=np.load(op)
        nr=next(x for x in newrecords if x['view']==view and x['fold']==f);npth=HEAVY/'sensor-models'/nr['context'];assert s.sha(npth)==rec['source_new_context_sha256'];nc=np.load(npth)
        assert src['train_ids']==ids[tr].tolist() and src['test_ids']==ids[te].tolist()
        file=HEAVY/'refinement'/rec['context'];assert s.sha(file)==rec['context_sha256'];ctx=np.load(file)
        A=np.column_stack([oc['A'],nc['A']]);T=np.column_stack([oc['T'],nc['T']]);np.testing.assert_array_equal(A,ctx['A']);np.testing.assert_array_equal(T,ctx['T'])
        mp=HEAVY/'refinement'/rec['model'];assert s.sha(mp)==rec['model_sha256'];models=joblib.load(mp)
        if view not in oldpred:oldpred[view]=np.full(len(y),np.nan)
        oldpred[view][te]=src['predictions']['stacked']
        for name,m in models.items():
            if name.startswith('ridge'):
                center=A.mean(0);scale=np.maximum(A.std(0),1e-6);ym=y[tr].mean();z=(A-center)/scale
                np.testing.assert_allclose(m['center'],center,atol=1e-12,rtol=0);np.testing.assert_allclose(m['scale'],scale,atol=1e-12,rtol=0);assert abs(m['ymean']-ym)<1e-12
                grad=z.T@(z@m['coef']-(y[tr]-ym))+m['alpha']*m['coef'];positive=m['coef']>1e-10
                assert np.max(abs(grad[positive]),initial=0)<1e-6 and np.min(grad[~positive],initial=0)>-1e-6
                pp=np.maximum(.1,(T-center)/scale@m['coef']+ym)
            else:
                w=m['weights'];assert (w>=0).all() and abs(w.sum()-1)<1e-8
                wt=np.where(y[tr]>=20,m['tail'],np.where(y[tr]>=12,np.sqrt(m['tail']),1.));wt/=wt.sum()
                objective=float(np.sum(wt*abs(A@w+m['bias']-y[tr]))+.01*abs(w-1/A.shape[1]).sum())
                assert abs(objective-m['objective'])<1e-7
                independently_refit=r.fit_lp(A,y[tr],m['tail']);assert abs(independently_refit['objective']-objective)<1e-7
                pp=np.maximum(.1,T@w+m['bias'])
            key=view+'/'+name
            if key not in pred:pred[key]=np.full(len(y),np.nan)
            pred[key][te]=pp;dd=float(abs(pp-stored[key.replace('/','__')][te]).max());assert dd<1e-9;maxdiff=max(maxdiff,dd);count+=1
        print('REFINEMENT_VERIFIED',view,f,flush=True)
    assert count==75
    summary={k:metric(y,z) for k,z in pred.items()};comp={}
    for k,z in pred.items():
        view=k.split('/')[0];policy=k.split('/')[1]
        comp[k]={'previous_same_features':delta(abs(z-y),abs(oldpred[view]-y)),
          'matched_speed':delta(abs(z-y),abs(pred['speed/'+policy]-y)),
          'previous_metric12':delta(abs(z-y),abs(oldpred['metric12']-y))}
    s.save(HERE/'refinement-results.json',{'summary':summary,'comparisons':comp,'primary':'ridge10 fixed before fitting; everyother method exploratory secondary',
      'limits':'Original sensor-derived values, original development594 reused; optional-feature missingness may encode original IMU detector quality. No actual camera accuracy claim.'})
    s.save(HERE/'refinement-verification.json',{'passed':True,'meta_models_replayed':count,'lp_models_refitted':30,'source_context_pairs_checked':15,
      'max_replay_abs':maxdiff,'results_sha256':s.sha(HERE/'refinement-results.json'),'protocol_sha256':s.sha(HERE/'refinement-protocol.json')})
    print('REFINEMENT_VERIFICATION_PASSED',flush=True)
if __name__=='__main__':main()
