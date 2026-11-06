from study import *
import math
def main():
    predictions={c:pd.read_csv(HERE/c/'predictions.csv') for c in ['exclude283','include283']}
    for r in json.loads((HERE/'stability.json').read_text()):
        p=predictions[r['scenario']];z=p[(p.id==r['id'])&(p.view==r['view'])].sort_values('repeat');v=list(z.prediction);mean=sum(v)/5
        sd=math.sqrt(sum((x-mean)**2 for x in v)/5)
        assert abs(mean-r['mean_prediction'])<1e-10 and abs(sd-r['prediction_SD'])<1e-10
        assert abs(min(v)-r['min_prediction'])<1e-10 and abs(max(v)-r['max_prediction'])<1e-10 and v==r['predictions_by_partition']
        e=np.array(v)-z.target.to_numpy()
        for k,vv in [('MAE',abs(e).mean()),('underprediction',np.maximum(-e,0).mean()),('overprediction',np.maximum(e,0).mean())]:assert abs(vv-r[k])<1e-10
    def persons(scenario,view,band):
        p=predictions[scenario];z=p[(p.view==view)&(p.id!='sub-283')].copy()
        if band=='ge20':z=z[z.target>=20]
        e=z.prediction-z.target;z['MAE']=abs(e);z['underprediction']=np.maximum(-e,0);z['overprediction']=np.maximum(e,0);z['bias']=e
        return z.groupby('id')[['MAE','underprediction','overprediction','bias']].mean().sort_index()
    s=json.loads((HERE/'common176-inclusion-effect.json').read_text())
    for r in s['metrics']:
        if r['band']=='lt20':continue
        z=persons(r['scenario'],r['view'],r['band']);assert len(z)==r['people']
        for metric in z:assert abs(z[metric].mean()-r[metric])<1e-10
    for r in s['contrasts']:
        a=persons(r['a'][0],r['a'][2],r['band']);b=persons(r['b'][0],r['b'][2],r['band']);assert a.index.equals(b.index)
        ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
        for metric in a:
            delta=(a[metric]-b[metric]).to_numpy();assert abs(delta.mean()-r[metric]['difference'])<1e-10
            assert np.allclose(np.quantile(delta[ix].mean(1),[.025,.975]),r[metric]['conditional_bootstrap95'],atol=1e-10,rtol=0)
    for r in json.loads((HERE/'exclude397-evaluation-only.json').read_text())['metrics']:
        p=predictions[r['scenario']];z=p[(p.view==r['view'])&(p.target>=20)&~p.id.isin(['sub-283','sub-397'])]
        assert z.id.nunique()==7;e=z.prediction-z.target
        for k,v in [('MAE',abs(e).mean()),('underprediction',np.maximum(-e,0).mean()),('overprediction',np.maximum(e,0).mean())]:assert abs(v-r[k])<1e-10
    save('stability-verification.json',{'passed':True,'case397_and283_SD_range_and_errors':True,'matched176_inclusion_metrics_and_intervals':True,'evaluation_only_leave397':True})
    print('STABILITY AND MATCHED-COHORT INCLUSION EFFECT VERIFIED',flush=True)
if __name__=='__main__':main()
