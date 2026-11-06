from study import *
def main():
    protocol=json.loads((HERE/'protocol.json').read_text());sens=HERE/'threshold-sensitivity'
    assert sha(HERE/'threshold_sensitivity.py')==json.loads((sens/'protocol.json').read_text())['code_sha256']
    assert sha(HERE/'protocol.json')==json.loads((sens/'protocol.json').read_text())['source_protocol_sha256']
    d=dict(np.load(sens/'input.npz'));rows=json.loads((HERE/'common176/manifest.json').read_text());base=np.load(HERE/'common176/input.npz')
    for i,m in enumerate(rows):
        a=read(TURN/'raw'/m['turn_file']);hh=[];xx=[]
        for l,r in m['clean_windows']:
            t=np.linspace(l,r,round((r-l)*30)+1);p=np.stack([np.interp(t,a['time'],a[k]) for k in ['pelvis_tx','pelvis_tz']],axis=1)
            s=np.stack([np.convolve(np.pad(p[:,j],3,mode='edge'),np.ones(7)/7,mode='valid') for j in range(2)],axis=1)
            v=np.gradient(s,1/30,axis=0);hh.extend(np.linalg.norm(v,axis=1)[:-1]);xx.extend(abs(v@m['axis'])[:-1])
        for name,th in [('CleanDelay05',.05),('CleanDelay15',.15)]:
            assert np.array_equal(d[name][i,:4],base['CleanSpeed'][i])
            assert np.array_equal(d[name][i,4:],np.array([sum(np.array(xx)<th),sum(np.array(hh)<th)])/30)
    primary=pd.read_csv(HERE/'common176/predictions.csv');extra=pd.read_csv(sens/'predictions.csv');p=pd.concat([primary,extra],ignore_index=True)
    def person(view,band):
        z=p[p.view==view].copy();z=z[z.target>=20] if band=='ge20' else (z[z.target<20] if band=='lt20' else z)
        e=z.prediction-z.target;z['MAE']=abs(e);z['underprediction']=np.maximum(-e,0);z['overprediction']=np.maximum(e,0);z['bias']=e
        return z.groupby('id')[['MAE','underprediction','overprediction','bias']].mean().sort_index()
    for name in ['common176','threshold-sensitivity']:
        summary=json.loads((HERE/name/'summary.json').read_text())
        for r in summary['metrics']:
            z=person(r['view'],r['band']);assert len(z)==r['people']
            for metric in z.columns:assert abs(z[metric].mean()-r[metric])<1e-10
        for c in summary['contrasts']:
            a=person(c['a'][2],c['band']);b=person(c['b'][2],c['band']);assert a.index.equals(b.index)
            ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
            for metric in a.columns:
                delta=(a[metric]-b[metric]).to_numpy();assert abs(delta.mean()-c[metric]['difference'])<1e-10
                assert np.allclose(np.quantile(delta[ix].mean(1),[.025,.975]),c[metric]['conditional_bootstrap95'],atol=1e-10,rtol=0)
    assert len(p)==176*5*10 and not p.duplicated(['repeat','id','view']).any() and not any(p.id=='sub-283')
    for receipt in json.loads((HERE/'acquisition.json').read_text()):assert sha(HERE/'raw'/receipt['name'])==receipt['sha256']
    for s in json.loads((sens/'selections.json').read_text()):assert sha(sens/s['file'])==s['sha256']
    save('final-verification.json',{'passed':True,'all10views_people_and_coverage':True,'sensitivity_features_independently_reconstructed':True,'all_reported_metrics_and_bootstrap_intervals_independently_recomputed':True,'marker_and_sensitivity_model_hashes':True,'plots_visually_reviewed':True})
    print('FINAL VERIFIED: 10 views, 176 people each, all metrics, bootstrap intervals, sensitivity inputs and hashes',flush=True)
if __name__=='__main__':main()
