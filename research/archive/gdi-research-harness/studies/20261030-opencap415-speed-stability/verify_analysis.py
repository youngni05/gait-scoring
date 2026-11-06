"""Recalculate report statistics from complete held-out prediction matrices."""
from common import *
import pandas as pd

def main():
    verification=json.loads((HERE/'verification.json').read_text())
    assert verification['passed'] and verification['CPU_models_replayed']==400 and verification['CNN_models_replayed']==75
    d=load();cpu=pd.read_csv(HERE/'cpu-predictions.csv');cnn=pd.read_csv(HERE/'cnn-predictions.csv')
    pred=pd.concat([cpu,cnn],ignore_index=True);summary=json.loads((HERE/'summary.json').read_text())
    person=pd.read_csv(HERE/'person-error-analysis.csv').sort_values('sample')
    matrices={}
    for name in VIEWS:
        p=cpu[(cpu.model=='RF')&(cpu.view==name)];a=np.empty((15,298))
        for repeat in range(5):
            for j,seed in enumerate(SEEDS):
                g=p[(p.repeat==repeat)&(p.seed==seed)].sort_values('sample')
                assert len(g)==298;a[repeat*3+j]=g.prediction.to_numpy()
        matrices[name]=a
    full_error=abs(matrices['full']-d['y'][None,:]);mean_error=full_error.mean(0)
    assert np.allclose(person.mean_abs_error,mean_error,atol=1e-10,rtol=0)
    assert np.allclose(person.mean_signed_error,(matrices['full']-d['y'][None,:]).mean(0),atol=1e-10,rtol=0)
    assert np.allclose(person.prediction_std,matrices['full'].std(0,ddof=1),atol=1e-10,rtol=0)
    top=np.lexsort((np.arange(298),-mean_error))[:60]
    assert set(person.loc[person.high_error,'sample'])==set(top)
    labels=pd.read_csv(SOURCE/'raw/participant_info.csv').set_index(['subid','visit'])
    for _,r in person.iterrows():
        source=labels.loc[(r.id,r.visit)]
        assert r.diag==source.diag and r.sex==source.sex and r.ambulation_status==source.ambulation_status
        assert np.isclose(r.target,source.tft_tug) and np.isclose(r.age,source.age,equal_nan=True)
    baseline=np.stack([cpu[(cpu.model=='Median')&(cpu.repeat==repeat)].sort_values('sample').prediction.to_numpy() for repeat in range(5)])
    for key,b in [('mean_speed',matrices['mean_speed']),('speed_summary',matrices['speed_summary']),('pelvis_motion',matrices['pelvis_motion']),('Median',baseline)]:
        delta=mean_error-abs(b-d['y'][None,:]).mean(0)
        result=summary['paired_comparisons']['full_RF_minus_'+key]
        assert np.isclose(delta.mean(),result['mean_difference_seconds'],atol=1e-12)
        rng=np.random.default_rng(20261030);means=delta[rng.integers(0,298,(10000,298))].mean(1)
        assert np.allclose(np.quantile(means,[.025,.975]),result['conditional_person_bootstrap95'],atol=1e-12)
    for item in summary['models']:
        g=pred[(pred.view==item['view'])&(pred.model==item['model'])]
        metrics=[]
        for (rep,seed),rows in g.groupby(['repeat','seed']):
            assert len(rows)==298
            metrics.append(np.mean(abs(rows.prediction.to_numpy()-rows.target.to_numpy())))
        assert len(metrics)==item['runs'] and np.isclose(np.mean(metrics),item['mean_MAE'],atol=1e-12)
        assert np.isclose(min(metrics),item['min_MAE']) and np.isclose(max(metrics),item['max_MAE'])
    for item in summary['error_groups']:
        mask=person[item['group']].astype(str)==item['value'];g=person[mask]
        assert len(g)==item['people']
        assert np.isclose(g.mean_abs_error.mean(),item['full_RF_MAE'])
        assert np.isclose(g.mean_signed_error.mean(),item['mean_signed_error'])
        assert int(g.high_error.sum())==item['high_error_people']
    for item in summary['high_error_by_TUG']:
        g=person[person.high_error&(person.TUG_band==item['TUG_band'])]
        assert len(g)==item['people'] and np.isclose(g.mean_abs_error.mean(),item['MAE'])
        assert np.isclose(g.mean_signed_error.mean(),item['signed_error'])
    save('analysis-verification.json',{'passed':True,'people':298,'RF_repetitions':15,'CNN_repetitions':15,'high_error_people':60,'no_seed_ensemble_scoring':True,'all_reported_metrics_recalculated':True,'paired_person_bootstrap_recalculated':True,'metadata_join_rechecked':True})
    print('ANALYSIS_INDEPENDENT_RECALCULATION_PASSED')

if __name__=='__main__':main()
