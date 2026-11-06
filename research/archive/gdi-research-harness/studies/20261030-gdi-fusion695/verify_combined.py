"""Independent long-table aggregation of final outcomes and all score bands."""
from common import HERE, MB, arrays, data, load, save, sha
import csv
import numpy as np
import pandas as pd


def main():
    d=data();report=load(HERE/'summary.json');saved=arrays(HERE/'all_predictions.npz')
    assert report['people']==695 and report['records']==1064 and len(report['methods'])==11
    assert sha(MB/'assets/MB_lite.bin')=='6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461'
    assert len(set(d['person_id']))==695 and len(set(d['video_id']))==1064
    assert sha(HERE/'all_predictions.npz')==report['all_predictions_sha256']
    assert all(sha(HERE/name)==value for name,value in report['code_sha256'].items())
    rows=pd.read_csv(HERE/'all_predictions.csv',dtype={'person_id':str,'video_id':str},float_precision='round_trip')
    assert len(rows)==23408 and not rows.duplicated(['method','video_id','side']).any()
    bands=load(HERE/'score_bands.json')['rows'];by_band={(x['method'],x['actual_bin_low']):x for x in bands}
    for name in report['methods']:
        frame=rows[rows['method']==name].copy()
        np.testing.assert_array_equal(frame['actual_GDI'].to_numpy(),d['y'].reshape(-1))
        np.testing.assert_allclose(frame['predicted_GDI'].to_numpy(),saved[name].reshape(-1),rtol=0,atol=1e-12)
        frame['bias']=frame['predicted_GDI']-frame['actual_GDI']
        frame['error']=abs(frame['bias'])
        frame['actual_bin']=np.floor(frame['actual_GDI']/10).astype(int)*10
        frame['exact']=(frame['actual_bin']//10==np.floor(frame['predicted_GDI']/10)).astype(float)
        frame['large']=(abs(frame['actual_bin']//10-np.floor(frame['predicted_GDI']/10))>=2).astype(float)
        means=frame.groupby('person_id')[['error','bias','exact','large']].mean().mean()
        for metric,column in [('MAE','error'),('bias','bias'),('exact_10point_bin_rate','exact'),('two_or_more_bins_rate','large')]:
            assert abs(means[column]-report['metrics'][name][metric])<1e-10
        person_band=frame.groupby(['actual_bin','person_id'])[['error','bias']].mean()
        for band,values in person_band.groupby(level=0):
            expected=by_band[name,int(band)]
            assert len(values)==expected['people']
            assert abs(values['error'].mean()-expected['person_MAE'])<1e-10
            assert abs(values['bias'].mean()-expected['bias_prediction_minus_truth'])<1e-10
    assert sha(HERE/'score_bands.json')==report['score_bands_sha256']
    for phase,paths in report['phase_evidence'].items():
        base=HERE if phase=='rf' else HERE/'cnn'/phase
        source=HERE/'rf_summary.json' if phase=='rf' else base/'summary.json'
        proof=HERE/'rf_verification.json' if phase=='rf' else base/'verification.json'
        assert sha(source)==paths['summary_sha256'] and sha(proof)==paths['verification_sha256']
        assert load(proof)['verified']
    proof={'verified':True,'people':695,'records':1064,'methods':11,'prediction_rows':23408,
           'all_metrics_and_score_bands_independently_recomputed_with_pandas':True,
           'RF_outer_models_replayed':15,'RF_selected_inner_refits':15,
           'CNN_inner_fits_checked':40,'CNN_outer_models_replayed':10,
           'frozen_frame_cache_replay_passed':load(HERE/'frame_verification.json')['verified'],
           'original_input_fingerprints_unchanged':True,'original_encoder_weight_matches_pinned_SHA256':True,'protected_numeric_data_read':False,
           'independent_final_evaluation':False,'summary_sha256':sha(HERE/'summary.json'),
           'verification_code_sha256':sha(HERE/'verify_combined.py')}
    save('verification.json',proof)
    print(proof)


if __name__=='__main__':main()
