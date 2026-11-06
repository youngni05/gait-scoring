"""Descriptive separation of change errors and available input-quality proxies.

No causal attribution, new pose extraction, model inference, training or selection.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from audit import HERE, ROOT, digest

STUDY=ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study'
RA=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/robust-aggregation692/cache')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def corr(x,y):
    x,y=np.asarray(x),np.asarray(y)
    return float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None

def main():
    base=read(HERE/'change-results.json');prediction_path=Path(base['frozen_prediction_path'])
    assert digest(prediction_path)==base['frozen_prediction_sha256']
    with np.load(prediction_path,allow_pickle=False) as z:
        data={k:z[k].copy() for k in ['person_id','video_id','fold','y','bagged','CNN_bag','STT_bag']}
    with np.load(STUDY/'cnn-multiwindow695/bundle/data.npz',allow_pickle=False) as z:
        orig=z['original_row_index'].copy()
        for k in ['person_id','video_id','fold','y']:np.testing.assert_array_equal(z[k],data[k])
    rows=read(HERE/'change-pairs.json');visits=read(HERE/'development-visits.json')
    visit_map={(r['person_id'],r['exam_id']):r for r in visits}
    video_index={v:i for i,v in enumerate(data['video_id'])}
    counts=Counter(r['person_id'] for r in rows);assert len(counts)==229 and len(rows)==581
    quality={};sources={str(prediction_path):digest(prediction_path)}
    used_indices=set()
    for r in rows:
        for key in ['lower_age_exam','higher_age_exam']:
            v=visit_map[r['person_id'],r[key]]['video_ids'];assert len(v)==1
            used_indices.add(video_index[v[0]])
    for i in sorted(used_indices):
        p=STUDY/'cnn-multiwindow695/records'/('%04d.json'%orig[i]);proof=read(p)
        assert proof['video_id']==data['video_id'][i] and proof['person_id']==data['person_id'][i]
        q=dict(canonical_missing_max=proof['missing_fraction_max_coordinate'],
               ambiguous_fraction_fullclip=sum(1<=f<=proof['raw_frames_used'] for f in proof['ambiguous_frames'])/proof['raw_frames_used'],
               windows=len(proof['eligible_sliding_window_starts']),
               raw_frames=proof['raw_frames_used'],canonical_fallback=proof['canonical_fallback'])
        sources[str(p)]=digest(p)
        rawpath=p.with_suffix('.npz');assert digest(rawpath)==proof['artifact_sha256']
        with np.load(rawpath,allow_pickle=False) as z:raw=z['source_raw_matched'].reshape(124,25,3)
        joints=[9,10,11,12,13,14,19,21,22,24]
        confidence=raw[:,joints,2]
        finite=confidence[np.isfinite(confidence)]
        q['canonical_mean_confidence']=float(finite.mean()) if len(finite) else None
        q['canonical_zero_or_missing_confidence_fraction']=float((~np.isfinite(confidence)|(confidence==0)).mean())
        quality[i]=q
    # Refit model window variability is a proxy, not the variance of the full bagged model.
    for family in ['CNN','STT']:
        for f in range(5):
            path=RA/family/('fold%d_refit.npz'%f);proof=read(path.with_suffix('.json'))
            assert digest(path)==proof['cache_sha256']
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['row_index'],np.arange(1061))
                for k in ['person_id','video_id','fold','y']:np.testing.assert_array_equal(z[k],data[k])
                raw=z['window_raw'];owner=z['window_record'];scale=abs(z['calibration'][0,0])
                for i in used_indices:
                    if data['fold'][i]!=f:continue
                    values=raw[owner==i]*scale
                    assert len(values)==quality[i]['windows']
                    quality[i][family+'_window_sd']=float(values.std(axis=0).mean())
            sources[str(path)]=proof['cache_sha256']
    enriched=[];categories=defaultdict(lambda:dict(raw_count=0,weight=0.0,error_contribution=0.0))
    component={k:dict(numerator=0.,denominator=0.,true_abs=0.,pred_abs=0.,mae=0.) for k in ['bagged','CNN_bag','STT_bag']}
    for r in rows:
        a,b=[video_index[visit_map[r['person_id'],r[k]]['video_ids'][0]] for k in ['lower_age_exam','higher_age_exam']]
        t=data['y'][b]-data['y'][a];p=data['bagged'][b]-data['bagged'][a]
        np.testing.assert_allclose(t,r['true_delta_lr'],atol=1e-10,rtol=0);np.testing.assert_allclose(p,r['predicted_delta_lr'],atol=1e-10,rtol=0)
        weight=1/(229*counts[r['person_id']]*2)
        cats=[]
        for side in range(2):
            if t[side]*p[side]<0:cat='direction_reversed'
            elif p[side]==0:cat='predicted_no_change'
            elif abs(p[side])<abs(t[side]):cat='correct_direction_smaller_magnitude'
            else:cat='correct_direction_equal_or_larger_magnitude'
            cats.append(cat);c=categories[cat];c['raw_count']+=1;c['weight']+=weight;c['error_contribution']+=weight*abs(t[side]-p[side])
        for k,m in component.items():
            d=data[k][b]-data[k][a]
            m['numerator']+=weight*float(t@d);m['denominator']+=weight*float(t@t)
            m['true_abs']+=weight*float(abs(t).sum());m['pred_abs']+=weight*float(abs(d).sum())
            m['mae']+=weight*float(abs(t-d).sum())
        q={k:max(quality[a][k],quality[b][k]) for k in ['canonical_missing_max','ambiguous_fraction_fullclip','canonical_zero_or_missing_confidence_fraction','CNN_window_sd','STT_window_sd']}
        enriched.append(dict(**r,index_a=int(a),index_b=int(b),category_lr=cats,
                             pair_mae=float(abs(t-p).mean()),mean_actual_abs_change=float(abs(t).mean()),
                             actual_initial_lr=data['y'][a].tolist(),actual_later_lr=data['y'][b].tolist(),
                             predicted_initial_lr=data['bagged'][a].tolist(),predicted_later_lr=data['bagged'][b].tolist(),
                             worst_visit_quality=q))
    for m in component.values():
        m['through_origin_delta_slope_descriptive']=m.pop('numerator')/m.pop('denominator')
        m['mean_absolute_predicted_change_divided_by_actual']=m['pred_abs']/m['true_abs']
    total_error=sum(c['error_contribution'] for c in categories.values())
    for c in categories.values():c['share_of_total_absolute_error']=c['error_contribution']/total_error
    assert abs(total_error-base['overall']['fixed_model']['person_mean_absolute_delta_error'])<1e-10
    people=defaultdict(list)
    for r in enriched:people[r['person_id']].append(r)
    person_rows=[]
    for person,rs in people.items():
        person_rows.append(dict(person_id=person,mae=float(np.mean([r['pair_mae'] for r in rs])),
                                actual_change=float(np.mean([r['mean_actual_abs_change'] for r in rs])),
                                **{k:float(np.mean([r['worst_visit_quality'][k] for r in rs])) for k in rs[0]['worst_visit_quality']}))
    associations={k:corr([r[k] for r in person_rows],[r['mae'] for r in person_rows]) for k in person_rows[0] if k not in ['person_id','mae']}
    top_people=sorted(person_rows,key=lambda r:r['mae'],reverse=True)[:10]
    cases=[]
    for n,pr in enumerate(top_people,1):
        r=max(people[pr['person_id']],key=lambda r:r['pair_mae'])
        cases.append(dict(case_id='C%02d'%n,person_mean_mae=pr['mae'],**r,
                          quality_initial=quality[r['index_a']],quality_later=quality[r['index_b']]))
    opposite=[r for r in enriched if r['true_delta_lr'][0]*r['true_delta_lr'][1]<0]
    lost=[r for r in opposite if r['predicted_delta_lr'][0]*r['predicted_delta_lr'][1]>=0]
    asymmetry=dict(actual_opposite_direction_pairs=len(opposite),predicted_same_direction_pairs=len(lost),
                   person_weighted_conditional_fraction=sum(1/counts[r['person_id']] for r in lost)/sum(1/counts[r['person_id']] for r in opposite),
                   interpretation='Different left/right change directions were not preserved; not proof of a left/right extraction swap')
    report=dict(people=229,pairs=581,unique_visits_checked=len(used_indices),side_pair_entries=1162,
                categories=dict(categories),component_diagnostics=component,
                left_right_change_diagnostics=asymmetry,
                person_level_descriptive_pearson_with_mae=associations,
                top_case_selection='10 people with largest person-mean error; worst pair per selected person',
                limitations=['Quality proxies are associations, not extraction-error ground truth or causal attribution',
                             'Raw coordinates and confidence refer to original matched124 frames; full-clip ambiguity is separate',
                             'Window SD uses the held-out refit constituent, not full bagged uncertainty or repeatability',
                             'No RGB, verified capture angle, per-video FPS, acquisition date or stride-synchronous GDI'],
                clinical_thresholds_or_model_parameters_changed=False,
                source_sha256=sources,script_sha256=digest(Path(__file__)))
    save(HERE/'cause-results.json',report);save(HERE/'cause-pairs.json',enriched);save(HERE/'cause-cases.json',cases)
    save(HERE/'cause-people.json',person_rows)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
    print(json.dumps([dict(case_id=r['case_id'],person_mae=r['person_mean_mae'],actual=r['true_delta_lr'],predicted=r['predicted_delta_lr'],missing=[r['quality_initial']['canonical_missing_max'],r['quality_later']['canonical_missing_max']],window_sd=[r['quality_initial']['CNN_window_sd'],r['quality_later']['CNN_window_sd']]) for r in cases],indent=2))

if __name__=='__main__':main()
