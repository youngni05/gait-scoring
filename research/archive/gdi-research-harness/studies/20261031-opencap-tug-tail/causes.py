"""Separate observed label/trajectory/model mechanisms from clinical hypotheses."""
from common import *
import pandas as pd
import joblib

def main():
    d=load();x=features(d)['raw'];y=d['y'];plan=json.loads((HERE/'splits.json').read_text())
    old={(r['repeat'],r['fold'],r['seed']):r for r in json.loads((HERE/'baseline-selections.json').read_text())}
    prev=pd.read_csv(PREV/'cpu-predictions.csv');prev=prev[(prev.model=='RF')&(prev.view=='speed_summary')]
    records=[];availability=[];weights_out=[];maxdiff=0.
    for rep in plan:
        for f in rep['splits']:
            tr=np.array(f['train']);te=np.array(f['test']);tail=te[y[te]>=20]
            availability.append({'repeat':rep['repeat'],'fold':f['fold'],'train_people':len(tr),'train_long':int((y[tr]>=20).sum()),'test_long':len(tail),'inner_train_long':[int((y[i['train']]>=20).sum()) for i in f['inner']],'inner_validation_long':[int((y[i['validation']]>=20).sum()) for i in f['inner']]})
            if not len(tail):continue
            for seed in SEEDS:
                r=old[(rep['repeat'],f['fold'],seed)];scale,model=joblib.load(PREV/r['model_path'])
                ztrain=scale.transform(x[tr]);ztest=scale.transform(x[tail]);influence=np.zeros((len(tail),len(tr)))
                for tree,bootstrap in zip(model.estimators_,model.estimators_samples_):
                    trainleaves=tree.apply(ztrain);testleaves=tree.apply(ztest)
                    counts=np.bincount(bootstrap,minlength=len(tr)).astype(float)
                    for j,leaf in enumerate(testleaves):
                        w=counts*(trainleaves==leaf);assert w.sum()>0
                        influence[j]+=w/w.sum()/len(model.estimators_)
                predictions=model.predict(ztest);reconstructed=influence@y[tr]
                assert np.allclose(reconstructed,predictions,atol=1e-9,rtol=0)
                maxdiff=max(maxdiff,float(np.max(abs(reconstructed-predictions))))
                for j,k in enumerate(tail):
                    dist=np.linalg.norm(ztrain-ztest[j],axis=1);near=np.argsort(dist)[:5]
                    stored=prev[(prev.repeat==rep['repeat'])&(prev.seed==seed)&(prev['sample']==k)].prediction.item()
                    assert np.isclose(stored,predictions[j],atol=1e-9)
                    records.append({'repeat':rep['repeat'],'seed':seed,'fold':f['fold'],'sample':int(k),'id':d['ids'][k],'target':float(y[k]),'prediction':float(predictions[j]),'train_long':int((y[tr]>=20).sum()),'train_target_max':float(y[tr].max()),'range_only_minimum_underprediction':float(max(0,y[k]-y[tr].max())),'long_label_weight_in_prediction':float(influence[j,y[tr]>=20].sum()),'nearest5_mean_TUG':float(y[tr[near]].mean()),'nearest5_long_count':int((y[tr[near]]>=20).sum()),'nearest5_distance_mean':float(dist[near].mean()),'outside_train_feature_dimensions':int(((x[k]<x[tr].min(0))|(x[k]>x[tr].max(0))).sum())})
                    weights_out.append({'repeat':rep['repeat'],'seed':seed,'fold':f['fold'],'sample':int(k),'train_indices':tr.tolist(),'weights':influence[j].tolist()})
    rows=pd.DataFrame(records);assert len(rows)==135 and rows.id.nunique()==9
    rows.to_csv(HERE/'baseline-tail-mechanisms.csv',index=False)
    save('forest-training-label-weights.json',weights_out);save('tail-availability.json',availability)
    man=pd.read_csv(PREV/'manifest.csv');labels=pd.read_csv(SOURCE/'raw/participant_info.csv').set_index(['subid','visit'])
    people=[];trajectories={}
    for k in np.flatnonzero(y>=20):
        r=man.iloc[k];clinical=labels.loc[(r.id,r.visit)]
        assert r.id==d['ids'][k] and np.isclose(clinical.tft_tug,y[k])
        path=SOURCE/'raw'/r.file;assert sha(path)==r.source_sha256
        lines=path.read_text().splitlines();h=lines.index('endheader');cols=lines[h+1].split();a=np.loadtxt(path,skiprows=h+2)
        grid=float(r.start_time)+np.arange(91)/30
        pos=np.array([np.interp(grid,a[:,0],a[:,cols.index(c)]) for c in ['pelvis_tx','pelvis_tz']])
        velocity=np.gradient(pos,1/30,axis=1);speed=np.hypot(velocity[0],velocity[1])
        assert np.allclose(speed,d['x'][k,17],atol=1e-5,rtol=1e-6)
        trajectories[str(r.id)]={'sample':int(k),'time_from_window_start':(grid-grid[0]).tolist(),'pelvis_x':pos[0].tolist(),'pelvis_z':pos[1].tolist(),'speed':speed.tolist()}
        g=rows[rows['sample']==k]
        people.append({'sample':int(k),'id':r.id,'visit':int(r.visit),'TUG':float(y[k]),'diag':clinical.diag,'clinical_10m_walk_seconds':float(clinical.tft_10mwt) if pd.notna(clinical.tft_10mwt) else None,'clinical_5xsts_seconds':float(clinical.tft_5xsts) if pd.notna(clinical.tft_5xsts) else None,'speed_mean':float(speed.mean()),'speed_median':float(np.median(speed)),'speed_max':float(speed.max()),'speed_IQR':float(np.quantile(speed,.75)-np.quantile(speed,.25)),'max_displacement_per_1over30_second':float(np.linalg.norm(np.diff(pos,axis=1),axis=0).max()),'source_max_dt':float(np.diff(a[:,0]).max()),'mean_prediction':float(g.prediction.mean()),'MAE':float((g.target-g.prediction).abs().mean()),'long_label_weight_mean':float(g.long_label_weight_in_prediction.mean()),'nearest5_mean_TUG':float(g.nearest5_mean_TUG.mean()),'feature_dimensions_outside_training_range_mean':float(g.outside_train_feature_dimensions.mean()),'range_only_minimum_underprediction_mean':float(g.range_only_minimum_underprediction.mean())})
    save('long-people-audit.json',people);save('long-trajectories.json',trajectories)
    summary={'people':298,'long_people':9,'long_predictions':135,'baseline_long_MAE':float(abs(rows.target-rows.prediction).mean()),'baseline_long_signed_error':float((rows.prediction-rows.target).mean()),'outer_train_long_range':[min(r['train_long'] for r in availability),max(r['train_long'] for r in availability)],'inner_train_long_range':[min(min(r['inner_train_long']) for r in availability),max(max(r['inner_train_long']) for r in availability)],'mean_long_label_weight_in_prediction':float(rows.long_label_weight_in_prediction.mean()),'median_long_label_weight_in_prediction':float(rows.long_label_weight_in_prediction.median()),'nearest5_mean_long_count':float(rows.nearest5_long_count.mean()),'range_limit_lower_bound_mean':float(rows.range_only_minimum_underprediction.mean()),'forest_weight_reconstruction_max_delta':maxdiff,'largest_TUG_person':max(people,key=lambda r:r['TUG']),'causal_limit':'These show data scarcity, feature mismatch and forest averaging/range mechanisms; cannot assign individual clinical causes or prove camera tracking error without original video/phase labels.'}
    save('cause-summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
