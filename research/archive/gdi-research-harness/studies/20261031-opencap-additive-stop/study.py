from pathlib import Path
import sys,json,hashlib,importlib.util
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261031-opencap-long-observation'
sys.path.insert(0,str(PRIOR))
spec=importlib.util.spec_from_file_location('long_study',PRIOR/'study.py');long=importlib.util.module_from_spec(spec);spec.loader.exec_module(long)
np,pd,read,sha=long.np,long.pd,long.read,long.sha
TURN=long.TURN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
ALPHAS=[.1,1.,10.,100.]
def save(n,v):
    p=HERE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def is_add(view):return view.startswith(('Add','Move'))
def fit(x,y,tr,te,alpha,view):
    additive=is_add(view);xx=x[:,:4] if additive else x
    offset=x[:,-1] if additive else np.zeros(len(y));target=y-offset
    scale=StandardScaler().fit(xx[tr]);model=Ridge(alpha=alpha,solver='svd').fit(scale.transform(xx[tr]),target[tr])
    rawpart=model.predict(scale.transform(xx[te]));raw=rawpart+offset[te]
    prediction=np.maximum(rawpart,0)+offset[te]
    return prediction,raw,(scale,model)

def moving_features(traces,record,th):
    t=traces['time'];mask=traces['valid'].copy()
    for e in record['events'][th]['horizontal']:
        lo=round((e['start']-t[0])*30);hi=round((e['end']-t[0])*30);mask[lo:hi]=False
    speed=traces['speed'][:-1];adj=mask[1:]&mask[:-1]
    assert mask.sum()>=15 and adj.any()
    return np.array([speed[mask].mean(),speed[mask].std(),abs(np.diff(speed))[adj].mean(),record['turn_seconds']],float),float(mask.sum()/30)

def prepare():
    assert not (HERE/'protocol.json').exists()
    oldrows=json.loads((PRIOR/'capped176/manifest.json').read_text());oldinput=np.load(PRIOR/'capped176/input.npz')
    original=json.loads((TURN/'geometry/manifest.json').read_text());m=next(r for r in original if r['id']=='sub-283');a=read(TURN/'raw'/m['turn_file'])
    assert sha(TURN/'raw'/m['turn_file'])==m['turn_sha256'];choice=long.old.choose(a,m);assert choice['status']=='passed'
    m={**m,'tug_walk_windows':choice['windows']};pre,post,extent=long.old.eligible(a,m)
    s={'id':m['id'],'start':pre[0] if pre else choice['windows'][0][0],'pre':pre,'post':post}
    records={r['id']:r for r in json.loads((PRIOR/'scout.json').read_text())};records['sub-283']=s
    fullrows=oldrows+[m];ids=np.array([m['id'] for m in fullrows]);y=np.array([m['target'] for m in fullrows]);views={};manifests=[]
    for b in [10,20,30]:
        for prefix in ['Speed','Learned10','Learned15','Add10','Add15','Move10','Move15']:views[f'{prefix}_{b}']=[]
    for i,m in enumerate(fullrows):
        a=read(TURN/'raw'/m['turn_file']);budgets={}
        for b in [10,20,30]:
            base,delays,r,traces=long.extract(a,m,records[m['id']],b)
            if i<176:
                assert np.array_equal(base,oldinput[f'Speed_{b}'][i])
                assert r==m['budgets'][str(b)]
            views[f'Speed_{b}'].append(base);moving_seconds={}
            for th in ['10','15']:
                pause=delays[th][2] # horizontal total only: axial stalls may be sideways motion.
                assert 0<=pause<y[i]
                views[f'Learned{th}_{b}'].append(np.r_[base,delays[th]])
                views[f'Add{th}_{b}'].append(np.r_[base,pause])
                moving,seconds=moving_features(traces,r,th);moving_seconds[th]=seconds
                views[f'Move{th}_{b}'].append(np.r_[moving,pause])
            budgets[str(b)]={**r,'moving_feature_seconds':moving_seconds}
        manifests.append({**m,'budgets':budgets})
    views={k:np.stack(v) for k,v in views.items()}
    originalids=[m['id'] for m in original];fullmap={originalids.index(id):i for i,id in enumerate(ids)}
    originalplans=json.loads((TURN/'geometry/splits.json').read_text());configs={}
    for cohort,ix in [('exclude283',np.arange(176)),('include283',np.arange(177))]:
        out=HERE/cohort;out.mkdir(exist_ok=True);np.savez_compressed(out/'input.npz',ids=ids[ix],y=y[ix],**{k:v[ix] for k,v in views.items()});save(cohort+'/manifest.json',[manifests[i] for i in ix])
        mapping={old:new for old,new in fullmap.items() if new in ix};remap=lambda xs:[mapping[j] for j in xs if j in mapping]
        plans=[{**rep,'splits':[{'fold':f['fold'],'train':remap(f['train']),'test':remap(f['test']),'inner':[{'train':remap(v['train']),'validation':remap(v['validation'])} for v in f['inner']]} for f in rep['splits']]} for rep in originalplans]
        # Keep order exactly as previous176, append283 where its original membership belongs.
        previous=json.loads((PRIOR/'capped176/splits.json').read_text())
        for rep,pr in zip(plans,previous):
            for f,pf in zip(rep['splits'],pr['splits']):
                for key in ['train','test']:
                    assert [i for i in f[key] if i!=176]==pf[key]
                    f[key]=pf[key]+([176] if 176 in f[key] else [])
                for v,pv in zip(f['inner'],pf['inner']):
                    for key in ['train','validation']:
                        assert [i for i in v[key] if i!=176]==pv[key]
                        v[key]=pv[key]+([176] if 176 in v[key] else [])
        save(cohort+'/splits.json',plans)
        configs[cohort]={'people':len(ix),'long_people':int(sum(y[ix]>=20)),'views':list(views),'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json']}}
    sources=[PRIOR/'capped176'/n for n in ['input.npz','manifest.json','splits.json','predictions.csv']]+[PRIOR/'study.py',PRIOR/'scout.py',TURN/'geometry/manifest.json',TURN/'geometry/splits.json']
    protocol={'cohorts':configs,'primary':'exclude283, maximum20s, Add10 versus Learned10 and Speed; Move10 secondary moving-only input; all10/20/30 and0.15 sensitivity reported',
      'additive':'training target=clinical TUG-observed horizontal low-speed episode total; Ridge on same4 speed/turn inputs; final=max(predicted remainder,0)+observed episode total; offset coefficient fixed1; no maximum clipping',
      'moving_only':'Move removes those same horizontal stop episodes from speed mean/std/within-contiguous-interval differences; same observed window, no extra recording; turn unchanged; residual target and fixed addition identical',
      'controls':'Speed predicts whole TUG from4features; Learned predicts whole TUG from4+4delay features with unconstrained coefficients; previous inputs exactly reproduced for176',
      'semantics':'remainder includes rise/sit and unobserved pauses, NOT independently labelled pure moving time; automatic low-speed episodes are stop candidates, not RGB-confirmed clinical pauses',
      'selection':'same alphas0.1/1/10/100, train-only scaler, inner finalTUG MAE; fixed offset measured from input, no evaluation target access',
      'split':'exclude176 byte-equivalent old membership/order; include177 restores original sub283 memberships in outer+inner folds; all other people unchanged; no random reshuffle',
      'observation':'exact previous nested windows and QC for176, same10/20/30 caps; same algorithm for283; no padding; caps are not fixed observed duration',
      'metrics':'person-equal MAE/under/over/bias over5 held-out partitions, >=20 subgroup8/9; conditional pairedperson bootstrap10000seed20261031; sub397 prediction populationSD and range across5partitions',
      'inclusion_analysis':'within each cohort compare models; include-vs-exclude also evaluate same176 and same8long; report283 separately; avoid interpreting subgroup9 vs8 as pure inclusion effect',
      'source_hashes':{str(p.relative_to(HERE.parent)):sha(p) for p in sources},'code_sha256':sha(HERE/'study.py')}
    save('protocol.json',protocol)
    save('case283-quality.json',{'selection':choice,'manifest':manifests[-1],'reason_for_previous_exclusion':'prior user-requested influence sensitivity; original separate-walk initial jump is not a blanket exclusion of all TUG recordings','passes_same_eligibility':True,'RGB_validation':False})
    print(json.dumps({'cohorts':{k:[v['people'],v['long_people'],len(v['views'])] for k,v in configs.items()},'case283_budgets':manifests[-1]['budgets']}),flush=True)
if __name__=='__main__':prepare()
