"""Re-read clinical/source files, audit splits, and replay every saved final model."""
from common import *
import io, joblib

def main():
    audit=json.loads((HERE/'preparation.json').read_text()); raw_checked=set();window_checks=0
    assert sha(HERE/'protocol.json')==audit['protocol_sha256']
    assert sha(HERE/'common.py')==audit['feature_code_sha256']
    for path,digest in audit['source_hashes'].items():assert sha(path)==digest,path
    labels=pd.read_csv(LABELS).set_index(['subid','visit']); assert labels.index.is_unique
    original=np.load(STABILITY/'input.npz')['ids'];old=json.loads((STABILITY/'splits.json').read_text())
    channels=json.loads((SOURCE/'all3s/protocol.json').read_text())['channels'][:17]
    datasets={}; split_checks=0;models=0;inner_checked=0;maxdelta=0.
    for name,meta in audit['cohorts'].items():
        d=load_data(name); datasets[name]=d;ids=d['ids'];n=len(ids)
        assert len(set(ids))==n==meta['people'] and 'sub-283' not in ids
        assert sha(HERE/name/'input.npz')==meta['input_sha256']
        assert sha(HERE/name/'splits.json')==meta['split_sha256']
        man=pd.read_csv(HERE/name/'manifest.csv');seconds=int(name[-1])
        assert list(ids)==man.id.tolist() and np.all(man.seconds==seconds)
        for i,r in enumerate(man.itertuples()):
            col='tft_5xsts' if name.startswith('STS') else 'tft_tug'
            assert float(labels.loc[(r.id,r.visit),col])==d['y'][i]==r.label
            assert np.array_equal(speed_features(d['x'][i]),d['speed'][i])
            np.testing.assert_allclose(patterns(d['x'][i])[0],d['pattern'][i,3:],equal_nan=True,atol=1e-12)
            assert np.array_equal(d['pattern'][i,:3],d['speed'][i])
            raw=SOURCE/'raw'/r.file
            if raw not in raw_checked:assert sha(raw)==r.source_sha256;raw_checked.add(raw)
            # Independent pandas read and explicit resampling of native coordinates.
            frame=pd.read_csv(io.StringIO(raw.read_text().split('endheader',1)[1].strip()),sep=r'\s+')
            t=frame.time.to_numpy();start=int(r.start_frame);end=np.searchsorted(t,t[start]+seconds)
            assert t[start]==r.start_time or abs(t[start]-r.start_time)<1e-7
            b=frame.iloc[start:end+1];grid=t[start]+np.arange(30*seconds+1)/30
            angles=np.array([np.interp(grid,b.time,b[c]) for c in channels])
            pos=np.array([np.interp(grid,b.time,b[c]) for c in ['pelvis_tx','pelvis_ty','pelvis_tz']])
            vel=np.empty_like(pos);vel[:,1:-1]=(pos[:,2:]-pos[:,:-2])*15
            vel[:,0]=(pos[:,1]-pos[:,0])*30;vel[:,-1]=(pos[:,-1]-pos[:,-2])*30
            rebuilt=np.vstack([angles,np.hypot(vel[0],vel[2]),vel[1]]).astype(np.float32)
            np.testing.assert_allclose(rebuilt,d['x'][i],rtol=1e-5,atol=1e-5);window_checks+=1
        plan=json.loads((HERE/name/'splits.json').read_text())
        for rep in plan:
            tested=[]
            for s,os in zip(rep['splits'],old[rep['repeat']]['splits']):
                tr,te=set(s['train']),set(s['test']);assert not tr&te and tr|te==set(range(n));tested+=s['test']
                assert set(ids[s['test']])==set(original[os['test']])&set(ids)
                validated=[]
                for inn,oi in zip(s['inner'],os['inner']):
                    it,iv=set(inn['train']),set(inn['validation'])
                    assert not it&iv and it|iv==tr and not (it|iv)&te
                    assert set(ids[inn['validation']])==set(original[oi['validation']])&set(ids)
                    validated+=inn['validation'];split_checks+=1
                assert sorted(validated)==sorted(tr)
            assert sorted(tested)==list(range(n))
    for outcome in ['STS','TUG']:
        a,b=datasets[outcome+'_common3'],datasets[outcome+'_common6']
        assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['y'],b['y'])
        # Last derivative sample differs because each window has its own endpoint.
        np.testing.assert_allclose(a['x'][:,:17],b['x'][:,:17,:91],rtol=0,atol=0)
        np.testing.assert_allclose(a['x'][:,17:,:-1],b['x'][:,17:,:90],rtol=0,atol=0)
    t=np.arange(181)/30;test=np.zeros((19,181));test[5]=30+20*np.cos(2*np.pi*t/1.2);test[11]=30+20*np.cos(2*np.pi*t/1.2+np.pi)
    v,q=patterns(test);assert abs(v[0]-1.2)<1e-8 and v[1]<1e-8 and q['cycle_available']
    flat,q=patterns(np.zeros((19,91)));assert np.isnan(flat[0]) and not q['cycle_available']
    jump=test.copy();jump[5,90]+=100;_,q=patterns(jump);assert q['bad_fraction']>0
    report={'source_hashes_checked':len(audit['source_hashes']),'raw_files_checked':len(raw_checked),
            'windows_reconstructed':window_checks,'inner_split_checks':split_checks,
            'same_common_people_start_and_seconds':True,'synthetic_cycle_flat_jump_checks':True}
    save('input-verification.json',dict(report,passed=True))
    if '--inputs-only' in sys.argv:print(json.dumps(report),flush=True);return
    for name in audit['cohorts']:
        d=datasets[name];pred=pd.read_csv(HERE/name/'predictions.csv');plan=json.loads((HERE/name/'splits.json').read_text())
        completion=json.loads((HERE/name/'completion.json').read_text())
        assert completion['predictions_sha256']==sha(HERE/name/'predictions.csv')
        assert completion['run_code_sha256']==sha(HERE/'run.py')
        selections=json.loads((HERE/name/'selections.json').read_text())
        assert len(selections)==completion['final_models']==200
        assert completion['inner_models']==1350
        for s in selections:
            split=plan[s['repeat']]['splits'][s['fold']];tr=np.array(split['train']);te=np.array(split['test'])
            scores=np.mean(np.abs(np.asarray(s['inner_predictions'])-d['y'][tr]),axis=1)
            np.testing.assert_allclose(scores,s['scores'],atol=1e-12)
            assert s['selected']==s['grid'][int(np.argmin(scores))];inner_checked+=len(scores)*3
            path=HERE/s['model_path'];assert sha(path)==s['model_sha256'];m=joblib.load(path);x=d[s['view']]
            imp=m[0];a=x[tr];expected=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in a.T])
            np.testing.assert_allclose(imp.statistics_,expected,atol=1e-12)
            # StandardScaler accumulates float32 inputs in float64.
            transformed=imp.transform(a).astype(np.float64)
            np.testing.assert_allclose(m[1].mean_,transformed.mean(0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(m[1].var_,transformed.var(0),rtol=1e-12,atol=1e-12)
            pp=m.predict(x[te]);got=pred[(pred.repeat==s['repeat'])&(pred.fold==s['fold'])&(pred.view==s['view'])&(pred.model==s['model'])&(pred.seed==s['seed'])].set_index('id')
            assert len(got)==len(te)
            delta=float(np.max(abs(pp-got.loc[d['ids'][te],'prediction'].to_numpy())));maxdelta=max(maxdelta,delta);assert delta<1e-8
            models+=1
        assert len(pred)==len(d['ids'])*5*9
        for (rep,model,view,seed),p in pred.groupby(['repeat','model','view','seed']):
            assert len(p)==len(d['ids']) and p.id.nunique()==len(d['ids'])
            if model=='Median':
                for s in plan[rep]['splits']:
                    v=np.sort(d['y'][s['train']])[(len(s['train'])-1)//2]
                    np.testing.assert_allclose(p[p.fold==s['fold']].prediction,v)
        print('REPLAY',name,'models total',models,flush=True)
    assert models==1200 and inner_checked==8100
    report.update(final_models_replayed=models,inner_fit_selections_checked=inner_checked,max_prediction_delta=maxdelta,passed=True)
    save('verification.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
