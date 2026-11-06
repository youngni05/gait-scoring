"""Gait-only phase pace addition, with prior fixed observed-stop offset."""
from pathlib import Path
import sys,json,importlib.util
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261031-opencap-additive-stop'
spec=importlib.util.spec_from_file_location('additive_source',PRIOR/'study.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
np,pd,read,sha=old.np,old.pd,old.read,old.sha
long=old.long
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
ALPHAS=[.1,1.,10.,100.]

def save(n,v):
    p=HERE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def fit(x,y,tr,te,alpha,view):
    xx=x[:,:-1];offset=x[:,-1]
    scale=StandardScaler().fit(xx[tr]);model=Ridge(alpha=alpha,solver='svd').fit(scale.transform(xx[tr]),y[tr]-offset[tr])
    remainder=model.predict(scale.transform(xx[te]))
    return np.maximum(remainder,0)+offset[te],remainder+offset[te],(scale,model)

def pace(traces,record,m):
    t=traces['time'];mask=traces['valid'].copy()
    for e in record['events']['10']['horizontal']:
        lo=round((e['start']-t[0])*30);hi=round((e['end']-t[0])*30);mask[lo:hi]=False
    speed=traces['speed'][:-1];mid=(t[:-1]+t[1:])/2
    assert mask.sum()>=15
    pooled=float(speed[mask].mean());assert pooled>0
    values=[];details={'moving_seconds':float(mask.sum()/30),'pooled_speed':pooled}
    for name,leg in [('outbound',mid<m['turn_start']),('return',mid>m['turn_end'])]:
        selected=mask&leg;seconds=float(selected.sum()/30)
        # Prespecified fallback preserves people if a leg has <0.5s moving support.
        fallback=seconds<.5
        mean=pooled if fallback else float(speed[selected].mean());assert mean>0
        values.append(1/mean);details[name]={'moving_seconds':seconds,'speed':mean,'fallback':fallback}
    return np.array([1/pooled]),np.array(values),details

def prepare():
    assert not (HERE/'protocol.json').exists()
    full=json.loads((PRIOR/'include283/manifest.json').read_text())
    scouts={r['id']:r for r in json.loads((long.HERE/'scout.json').read_text())}
    cached=np.load(long.HERE/'traces.npz');extra={};records=[];audit=[]
    for b in [10,20,30]:
        extra[f'Pooled_{b}']=[];extra[f'Directional_{b}']=[]
    data=np.load(PRIOR/'include283/input.npz')
    for i,m in enumerate(full):
        budgets={}
        for b in [10,20,30]:
            r=m['budgets'][str(b)]
            if m['id']!='sub-283':traces={k:cached[f'{m["id"]}_{b}_{k}'] for k in ['time','speed','valid']}
            else:
                a=read(long.TURN/'raw'/m['turn_file']);pre,post,extent=long.old.eligible(a,m)
                scout={'start':pre[0] if pre else m['tug_walk_windows'][0][0],'pre':pre,'post':post}
                _,_,rr,traces=long.extract(a,m,scout,b)
                assert rr['window']==r['window']
            pooled,directional,detail=pace(traces,r,m)
            base=data[f'Add10_{b}'][i]
            extra[f'Pooled_{b}'].append(np.r_[base[:-1],pooled,base[-1]])
            extra[f'Directional_{b}'].append(np.r_[base[:-1],directional,base[-1]])
            budgets[str(b)]=detail
            audit.append({'id':m['id'],'budget':b,**detail})
        records.append({**m,'pace':budgets})
    views={f'Baseline_{b}':data[f'Add10_{b}'].copy() for b in [10,20,30]}
    views.update({k:np.stack(v) for k,v in extra.items()})
    configs={}
    for cohort,n in [('exclude283',176),('include283',177)]:
        out=HERE/cohort;out.mkdir(exist_ok=True)
        np.savez_compressed(out/'input.npz',ids=data['ids'][:n],y=data['y'][:n],**{k:v[:n] for k,v in views.items()})
        save(cohort+'/manifest.json',records[:n]);(out/'splits.json').write_bytes((PRIOR/cohort/'splits.json').read_bytes())
        configs[cohort]={'people':n,'long_people':int(sum(data['y'][:n]>=20)),'views':list(views),'hashes':{p:sha(out/p) for p in ['input.npz','manifest.json','splits.json']}}
    sources=[PRIOR/c/n for c in configs for n in ['input.npz','manifest.json','splits.json','predictions.csv']]+[PRIOR/'study.py',long.HERE/'study.py',long.HERE/'traces.npz']
    save('protocol.json',{'cohorts':configs,'primary':'exclude283, max20s; Directional versus Baseline; Pooled ablation separates reciprocal pace from direction separation; all10/20/30 and include283 reported',
        'hypothesis':'whole-window speed hides slow outbound or return gait; separate moving pace seconds/m may explain residual TUG beyond observed gait-stop offset',
        'features':'Baseline exact prior Add10 (4 speed/turn columns + final fixed stop offset); Pooled adds 1/mean moving speed; Directional adds separate outbound/return 1/mean moving speed. Existing speed/turn columns and fixed offset unchanged.',
        'pace':'same raw horizontal speed as existing moving-speed comparison; remove existing horizontal <0.10m/s episodes >=0.5s with <=0.2s bridging; each direction at least0.5s else pooled-speed fallback; no clipping of reciprocal or prediction upper range',
        'exclusions':'no preparation, rise, sit, absolute timestamps, recording length or clinical target in new inputs; no person removal or new source acquisition',
        'model':'same train-only StandardScaler and Ridge SVD; alpha0.1/1/10/100 inner3fold finalTUG MAE; target TUG-stop; prediction max(residual,0)+stop;5 unchanged person partitions x5 outer folds',
        'interpretation':'pace is seconds per metre, not an observed or labelled whole-task movement duration; remainder also includes unobserved parts; stop candidates not independently video-confirmed',
        'metrics':'person-equal errors over5 partitions; MAE, under, over; >=20s8/9people; conditional pairedperson bootstrap10000 seed20261031; sub397 SD/range; evaluation-only removal of397; common176 comparison for inclusion sensitivity',
        'limits':'sameperson/visit clinical TUG but exact timed trial unconfirmed; full-record boundaries locate input windows, not online inference; maximum budgets not identical full-length10/20/30 acquisition; repeatedly inspected development cohort, no untouched external validation',
        'source_hashes':{str(p.relative_to(HERE.parent)):sha(p) for p in sources},'code_sha256':sha(HERE/'study.py')})
    save('pace-audit.json',audit)
    print(json.dumps({'cohorts':{k:[v['people'],v['long_people']] for k,v in configs.items()},'fallbacks':sum(r[k]['fallback'] for r in audit for k in ['outbound','return']),'minimum_speed':min(r[k]['speed'] for r in audit for k in ['outbound','return'])}),flush=True)
if __name__=='__main__':prepare()
