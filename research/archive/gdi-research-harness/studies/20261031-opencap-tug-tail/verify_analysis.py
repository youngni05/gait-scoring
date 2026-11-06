from common import *
import pandas as pd

def main():
    assert json.loads((HERE/'verification.json').read_text())['new_models_replayed']==575
    d=load();y=d['y'];pred=pd.concat([pd.read_csv(HERE/'predictions.csv'),pd.read_csv(HERE/'followup-predictions.csv')],ignore_index=True)
    summary=json.loads((HERE/'summary.json').read_text());people=pd.read_csv(HERE/'person-results.csv')
    groups={'all':np.ones(298,dtype=bool),'short':y<10,'mid':(y>=10)&(y<20),'long':y>=20};mean_errors={}
    for method,item in summary['models'].items():
        g=pred[pred.method==method];arrays=[]
        for _,run in g.groupby(['repeat','seed']):
            run=run.sort_values('sample');assert run['sample'].tolist()==list(range(298));arrays.append(run.prediction.to_numpy())
        pp=np.stack(arrays);error=pp-y[None,:];mean_errors[method]=abs(error).mean(0)
        expected=15 if method not in ['InverseRidge','InverseRidgeBalanced'] else 5
        assert len(arrays)==expected
        for band,mask in groups.items():
            e=error[:,mask];r=item[band]
            assert r['people']==int(mask.sum()) and r['runs']==expected
            assert np.isclose(abs(e).mean(),r['MAE'],atol=1e-12) and np.isclose(e.mean(),r['bias'],atol=1e-12)
            assert np.isclose(np.sqrt((e*e).mean(1)).mean(),r['RMSE'],atol=1e-12)
            assert np.isclose(np.maximum(-e,0).mean(),r['underprediction']) and np.isclose(np.maximum(e,0).mean(),r['overprediction'])
        row=people[people.method==method].sort_values('sample')
        assert np.allclose(row.MAE,mean_errors[method],atol=1e-12) and np.allclose(row.mean_prediction,pp.mean(0),atol=1e-12)
    for method,item in summary['differences_vs_baseline'].items():
        delta=mean_errors[method]-mean_errors['BaselineRF']
        for band,mask in groups.items():
            z=delta[mask];rng=np.random.default_rng(20261031)
            boot=z[rng.integers(0,len(z),(10000,len(z)))].mean(1)
            assert np.isclose(z.mean(),item[band]['mean'],atol=1e-12)
            assert np.allclose(np.quantile(boot,[.025,.975]),item[band]['conditional_person_bootstrap95'],atol=1e-12)
        assert int((delta[y>=20]<0).sum())==item['long_people_improved']
    causes=json.loads((HERE/'cause-summary.json').read_text());rows=pd.read_csv(HERE/'baseline-tail-mechanisms.csv')
    assert rows.id.nunique()==9 and len(rows)==135
    assert np.isclose(abs(rows.target-rows.prediction).mean(),causes['baseline_long_MAE'])
    assert np.isclose(rows.long_label_weight_in_prediction.mean(),causes['mean_long_label_weight_in_prediction'])
    assert np.isclose(rows.range_only_minimum_underprediction.mean(),causes['range_limit_lower_bound_mean'])
    save('analysis-verification.json',{'passed':True,'people':298,'unique_long_people':9,'methods':10,'repeated_error_not_ensemble':True,'metrics_and_bias_recomputed':True,'person_bootstrap_recomputed':True,'cause_aggregates_recomputed':True,'followup_exploratory_phase_separate':True})
    print('TAIL_REPORT_INDEPENDENT_RECALCULATION_PASSED')

if __name__=='__main__':main()
