from study import *
from threadpoolctl import threadpool_limits
from collections import defaultdict
import zlib
def main():
    protocol=json.loads((HERE/'protocol.json').read_text());results={};cases=json.loads((HERE/'case-predictions.json').read_text())
    for n,h in protocol['source_hashes'].items():assert sha(HERE.parent/n)==h
    index={x['name']:x for x in json.loads((HERE.parent/'20261030-opencap415-tug-timeseries/archive-index.json').read_text())}
    for r in json.loads((HERE/'acquisition.json').read_text()):
        p=(HERE/r['name']) if 'url' in r else (HERE/'raw'/r['name']);assert sha(p)==r['sha256']
        if 'crc_verified' in r:assert zlib.crc32(p.read_bytes())==index[r['name']]['crc']
    audit=json.loads((HERE/'case-timing-audit.json').read_text());previous={m['id']:m for m in json.loads((PRIOR/'include283/manifest.json').read_text())}
    trans={r['id']:r for r in json.loads((HERE.parent/'20261031-opencap-chair/transition-audit.json').read_text())}
    clinical=pd.read_csv(HERE.parent/'20261030-opencap415-tug-timeseries/raw/participant_info.csv')
    for r in audit:
        m=previous[r['id']];label=float(clinical[(clinical.subid==r['id'])&(clinical.visit==r['visit'])].iloc[0].tft_tug);assert label==r['target']==m['target']
        end=trans[r['id']]['sit']['window'][1];assert abs(end-m['phase']['extended'][0]-r['candidate_to_sit_seconds'])<1e-10;assert abs(end-m['phase']['core'][0]-r['core_to_sit_seconds'])<1e-10
        assert abs(end-label-r['label_derived_start_if_same_trial_and_sit_end_correct'])<1e-10
        assert r['actual_timer_start'] is None and r['actual_timer_end'] is None and not r['timed_trial_confirmed']
    for r in json.loads((HERE/'marker-boundaries.json').read_text()):
        fps,t,names,xyz=trc(HERE/r['file']);h=(xyz[:,list(names).index('RHJC_study'),1]+xyz[:,list(names).index('LHJC_study'),1])/2;h=(h-h.min())/(h.max()-h.min());b=r['author_height_proxy']
        if not b['identifiable']:assert not (any(h[:int(fps)]<.3) and any(h[-int(fps):]<.3));continue
        start=next(i for i,v in enumerate(h) if v>.3);last=max(i for i,v in enumerate(h) if v>.7);end=next(i for i in range(last+1,len(h)) if h[i]<.3)
        assert np.allclose([t[start],t[end],(end-start)/fps],[b['start'],b['end'],b['duration']],atol=1e-8,rtol=0)
    with threadpool_limits(limits=1):
        for c,info in protocol['cohorts'].items():
            out=HERE/c;d=dict(np.load(out/'input.npz'));y=d['y'];p=pd.read_csv(out/'predictions.csv');prior=pd.read_csv(PRIOR/c/'predictions.csv')
            plan={(r['repeat'],f['fold']):f for r in json.loads((out/'splits.json').read_text()) for f in r['splits']};inner=json.loads((out/'inner.json').read_text());selections=json.loads((out/'selections.json').read_text());keyed=defaultdict(list);maximum=0.
            for r in inner:
                f=plan[r['repeat'],r['fold']];full=d[r['view']];x=full[:,:-1];offset=full[:,-1];pred=np.full(len(y),np.nan)
                for v in f['inner']:
                    tr=np.array(v['train']);te=np.array(v['validation']);sc=StandardScaler().fit(x[tr]);m=Ridge(alpha=r['alpha'],solver='svd').fit(sc.transform(x[tr]),y[tr]-offset[tr]);pred[te]=np.maximum(m.predict(sc.transform(x[te])),0)+offset[te]
                assert r['train_order']==f['train'];diff=float(max(abs(pred[f['train']]-r['predictions'])));maximum=max(maximum,diff);assert diff<1e-10
                assert abs(np.mean(abs(pred[f['train']]-y[f['train']]))-r['MAE'])<1e-10;keyed[r['repeat'],r['fold'],r['view']].append(r)
            for s in selections:
                f=plan[s['repeat'],s['fold']];tr=np.array(f['train']);te=np.array(f['test']);x=d[s['view']][:,:-1];offset=d[s['view']][:,-1]
                best=min(keyed[s['repeat'],s['fold'],s['view']],key=lambda r:r['MAE']);assert best['alpha']==s['alpha'] and best['MAE']==s['inner_MAE']
                assert sha(HERE/s['file'])==s['sha256'];scale,model=joblib.load(HERE/s['file']);q=p[(p['repeat']==s['repeat'])&(p.fold==s['fold'])&(p.view==s['view'])].set_index('sample').loc[te]
                assert list(q.id)==list(d['ids'][te]) and np.array_equal(q.target,y[te])
                v=model.predict(scale.transform(x[te]));assert max(abs(v+offset[te]-q.raw_prediction))<1e-10;assert max(abs(np.maximum(v,0)+offset[te]-q.prediction))<1e-10
                sc=StandardScaler().fit(x[tr]);m=Ridge(alpha=s['alpha'],solver='svd').fit(sc.transform(x[tr]),y[tr]-offset[tr]);refit=np.maximum(m.predict(sc.transform(x[te])),0)+offset[te]
                diff=float(max(abs(refit-q.prediction)));maximum=max(maximum,diff);assert diff<1e-10
            assert len(p)==5*info['people']*len(info['views']) and not p.duplicated(['view','repeat','id']).any()
            for b in [10,20,30]:
                for a,z in [('Gait','Gait'),('Withheld','Reallocated'),('Candidate','Prep')]:
                    joined=p[p.view==f'{a}_{b}'].merge(prior[prior.view==f'{z}_{b}'],on=['repeat','fold','id'],suffixes=('_new','_prior'),validate='one_to_one');assert len(joined)==5*info['people'];assert max(abs(joined.prediction_new-joined.prediction_prior))<1e-10
            person=pd.read_csv(out/'person-results.csv');summary=json.loads((out/'summary.json').read_text())
            for r in person.itertuples():
                g=p[(p.view==r.view)&(p.id==r.id)];e=(g.prediction-g.target).to_numpy();assert np.allclose([abs(e).mean(),np.maximum(-e,0).mean(),np.maximum(e,0).mean(),e.mean()],[r.MAE,r.underprediction,r.overprediction,r.bias],atol=1e-10,rtol=0)
            def select(view,band):
                z=person[person.view==view].set_index('id').sort_index();return z[z.target>=20] if band=='ge20' else (z[z.target<20] if band=='lt20' else z)
            for r in summary['metrics']:
                g=select(r['view'],r['band']);assert len(g)==r['people']
                for k in ['MAE','underprediction','overprediction','bias']:assert abs(g[k].mean()-r[k])<1e-10
            for r in summary['contrasts']:
                a=select(r['a'][2],r['band']);b=select(r['b'][2],r['band']).loc[a.index];ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
                for k in ['MAE','underprediction','overprediction','bias']:
                    delta=(a[k]-b[k]).to_numpy();assert abs(delta.mean()-r[k]['difference'])<1e-10;assert np.allclose(np.quantile(delta[ix].mean(1),[.025,.975]),r[k]['conditional_bootstrap95'],atol=1e-10,rtol=0)
            for r in cases:
                if r['cohort']!=c:continue
                q=p[(p.view==r['view'])&(p.id==r['id'])].prediction.to_numpy();assert np.allclose([q.mean(),q.std(),q.min(),q.max(),abs(q-r['target']).mean()],[r[k] for k in ['mean','sd','min','max','MAE']],atol=1e-10,rtol=0)
            results[c]={'models':len(selections),'inner_fits':len(inner)*3,'max_difference':maximum,'all_prior_conditions_reproduced':True};print('VERIFIED',c,len(selections),flush=True)
    save('verification.json',{'passed':True,'cohorts':results,'models':sum(r['models'] for r in results.values()),'inner_fits':sum(r['inner_fits'] for r in results.values()),'raw_file_crc_sha':True,'timing_diagnostics_recomputed':True,'author_marker_function_reproduced':True,'actual_clinical_start_confirmed':False})
if __name__=='__main__':main()
