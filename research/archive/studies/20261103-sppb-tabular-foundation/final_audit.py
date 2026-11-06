"""Check final metrics against original saved predictions, independently of replay output."""
from pathlib import Path
import json, hashlib
import numpy as np
HERE=Path('D:/Projects/D_P/studies/20261103-sppb-tabular-foundation')
BIG=Path('D:/gait-research-data/D_P/sppb-tabular-foundation-20261103')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cohorts=read(BIG/'cohorts.json');comparisons={};verified=0
    for scope in ['small','large']:
        v=read(HERE/f'verification-{scope}.json');assert v['status']=='passed';assert sha(HERE/f'comparison-{scope}.json')==v['comparison_sha256'];comparisons.update(read(HERE/f'comparison-{scope}.json'));verified+=v['reconstructed_models']
    assert verified==165
    raw={};max_difference=0.0
    for file in (BIG/'predictions').glob('*.json'):
        r=read(file);name=r['cohort'];c=cohorts[name];ids=[p['id'] for p in c];ix={s:i for i,s in enumerate(ids)}
        key=r['condition']+'_'+r['kind'];raw.setdefault(name,{}).setdefault(key,np.full(len(ids),np.nan))[[ix[s] for s in r['test_ids']]]=r['prediction']
    for name,conditions in raw.items():
        y=np.array([p['sppb'] for p in cohorts[name]]);result=comparisons[name];assert result['ids']==[p['id'] for p in cohorts[name]]
        for key,p in conditions.items():
            assert np.isfinite(p).all();rp=np.array(result['predictions'][key]);max_difference=max(max_difference,float(np.max(abs(p-rp))));np.testing.assert_allclose(p,rp,atol=1e-5,rtol=1e-6)
            m=result['metrics'][key];error=p-y
            np.testing.assert_allclose([np.linalg.norm(error,ord=1)/len(y),np.linalg.norm(error)/np.sqrt(len(y)),sum(error)/len(y)],[m['MAE'],m['RMSE'],m['bias']],atol=1e-5,rtol=0)
            assert sum((y<=8)&(p>8))==m['FN'] and sum((y>8)&(p<=8))==m['FP']
    # Full-array bootstrap gives an independent check of the chunked calculation.
    c=cohorts['mobilised'];y=np.array([p['sppb'] for p in c]);idx=np.random.RandomState(20261103).randint(len(y),size=(10000,len(y)));ci_checks={}
    for kind in ['LGBM','TabICLv2','TabPFNv2']:
        d=abs(raw['mobilised']['features_'+kind]-y)-abs(raw['mobilised']['speed_profile_'+kind]-y);ci=np.percentile(np.mean(d[idx],axis=1),[2.5,97.5]);key=f'features_{kind} minus speed_profile_{kind}'
        np.testing.assert_allclose(ci,comparisons['mobilised']['contrasts'][key]['CI95'],atol=1e-5,rtol=0);ci_checks[key]=ci.tolist()
    audit=dict(status='passed',unique_new_models=verified,original_prediction_vs_replay_max_difference=max_difference,full_array_bootstrap_primary_CIs=ci_checks,code_sha256=sha(Path(__file__)),data_bytes=sum(p.stat().st_size for p in BIG.rglob('*') if p.is_file()),artifact_sha256={p.name:sha(p) for p in HERE.glob('*verification*.json')})
    (HERE/'final-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
