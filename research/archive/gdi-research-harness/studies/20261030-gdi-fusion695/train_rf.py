"""Nested RF representation comparison; outer outcomes never select settings."""
from common import HERE, data, errors, load, metrics, save, sha, split, summary
from datetime import datetime, timezone
import json
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def main():
    tick=time.monotonic()
    d=data(); protocol=load(HERE/'protocol.json')['RF']
    reps={'rf_coordinate':d['kinematic'],'rf_motionbert':d['motionbert'],
          'rf_concatenated':np.column_stack([d['kinematic'],d['motionbert']])}
    methods={name:np.full((1064,2),np.nan) for name in reps}
    selections=[]
    for name,x in reps.items():
        for outer in range(5):
            dest=HERE/'rf'/name/f'fold{outer}'
            dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():
                done=load(dest/'complete.json')
                assert sha(dest/'prediction.npz')==done['prediction_sha256']
                with np.load(dest/'prediction.npz',allow_pickle=False) as z: methods[name][z['indices']]=z['prediction']
                selections.append(done);continue
            tr,te=split(d,outer)
            inner=np.full((len(protocol['candidates']),len(tr),2),np.nan)
            for ci,candidate in enumerate(protocol['candidates']):
                for f in range(4):
                    it,iv=split(d,outer,f)
                    model=RandomForestRegressor(n_estimators=protocol['n_estimators'],**candidate,
                        random_state=protocol['seed']+outer*100+f,n_jobs=protocol['threads'])
                    model.fit(x[it],d['y'][it])
                    inner[ci,np.searchsorted(tr,iv)]=model.predict(x[iv])
            assert np.isfinite(inner).all()
            values=[float(errors(d['y'][tr],p,d['person_id'][tr]).mean()) for p in inner]
            chosen=min(i for i,v in enumerate(values) if v<=min(values)+1e-12)
            model=RandomForestRegressor(n_estimators=protocol['n_estimators'],**protocol['candidates'][chosen],
                random_state=protocol['seed']+outer*100+50,n_jobs=protocol['threads']).fit(x[tr],d['y'][tr])
            pred=model.predict(x[te]);methods[name][te]=pred
            np.savez_compressed(dest/'inner.npz',predictions=inner,train_indices=tr)
            np.savez_compressed(dest/'prediction.npz',prediction=pred,indices=te)
            joblib.dump(model,dest/'model.joblib',compress=3)
            done={'representation':name,'outer_fold':outer,'candidate_scores':values,'selected':chosen,
                  'parameters':protocol['candidates'][chosen], 'train_people':np.unique(d['person_id'][tr]).tolist(),
                  'evaluation_people':np.unique(d['person_id'][te]).tolist(),
                  'evaluation_MAE':float(errors(d['y'][te],pred,d['person_id'][te]).mean()),
                  'prediction_sha256':sha(dest/'prediction.npz'),'model_sha256':sha(dest/'model.joblib'),
                  'inner_sha256':sha(dest/'inner.npz'),'protocol_sha256':sha(HERE/'protocol.json')}
            save(dest.relative_to(HERE)/'complete.json',done);selections.append(done)
            print(json.dumps({'representation':name,'fold':outer,'selected':chosen,'MAE':done['evaluation_MAE'],
                              'elapsed_seconds':round(time.monotonic()-tick,1)}),flush=True)
    methods['pls']=d['pls']
    methods['rf_coordinate_pls_equal']=(methods['rf_coordinate']+d['pls'])/2
    methods['rf_concatenated_pls_equal']=(methods['rf_concatenated']+d['pls'])/2
    np.savez_compressed(HERE/'rf_predictions.npz',**methods)
    result={**summary(d,methods),'selected':selections,'elapsed_seconds':time.monotonic()-tick,
            'finished_utc':datetime.now(timezone.utc).isoformat(),'protected_numeric_data_read':False,
            'independent_final_evaluation':False,'total_RF_fits':255,
            'sha256':{name:sha(HERE/name) for name in ['train_rf.py','common.py','data.npz','protocol.json','rf_predictions.npz']}}
    save('rf_summary.json',result)
    print(json.dumps({'completed':'RF baselines and fusion','MAE':{k:v['MAE'] for k,v in result['metrics'].items()},
                      'seconds':result['elapsed_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
