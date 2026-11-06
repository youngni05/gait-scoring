from pathlib import Path
import sys,json,hashlib,importlib.util
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261031-opencap-leave283/geometry'
EXT=HERE.parent/'20261031-opencap-extrapolation'
spec=importlib.util.spec_from_file_location('old_fit',EXT/'study.py');oldfit=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldfit)
np,pd=oldfit.np,oldfit.pd
import joblib
ALPHAS=[.1,1.,10.,100.]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def fit(x,y,tr,te,alpha):return oldfit.fit(x,y,tr,te,'Ridge',alpha)

def pooled_speed(segments):
    # Keep 91 sample values and 90 within-leg differences for a total 90 intervals.
    # Never take a difference across the end/start of separate legs.
    joined=np.concatenate([v[:-1] for v in segments[:-1]]+[segments[-1]])
    changes=np.concatenate([abs(np.diff(v)) for v in segments])
    assert len(joined)==91 and len(changes)==90
    return np.array([joined.mean(),joined.std(),changes.mean()],dtype=np.float64)

def tug_features(a,windows):
    import scout
    from prepare import old
    speeds=[]
    for left,right in windows:
        index=int(np.searchsorted(a['time'],left-1e-7));seconds=right-left
        assert abs(float(a['time'][index])-left)<1e-6
        speeds.append(old.window(a,index,seconds)[17])
    return pooled_speed(speeds),speeds

def prepare():
    assert not (HERE/'protocol.json').exists()
    import scout
    d=dict(np.load(PRIOR/'input.npz'));rows=json.loads((PRIOR/'manifest.json').read_text());audit=json.loads((HERE/'scout.json').read_text())
    assert len(d['ids'])==237 and 'sub-283' not in d['ids'] and sum(d['y']>=20)==8
    base=d['TurnTimeOnly'].copy();assert base.shape==(237,4) and base[:,0].min()>.05
    inverse=base.copy();inverse[:,0]=1/base[:,0]
    keep=[];tug=[];records=[];traces={}
    for i,(m,r) in enumerate(zip(rows,audit)):
        assert m['id']==r['id']==d['ids'][i]
        if r['status']!='passed':continue
        path=scout.TURN/'raw'/m['turn_file'];assert sha(path)==m['turn_sha256'];a=scout.read(path)
        assert json.loads(json.dumps(scout.choose(a,m)))=={k:v for k,v in r.items() if k not in ['id','target']}
        feature,speeds=tug_features(a,r['windows']);keep.append(i);tug.append(np.r_[feature,base[i,3]])
        for j,v in enumerate(speeds):traces[f'{m["id"]}_{j}']=v
        records.append({**m,'source_sample':i,'tug_walk_windows':r['windows'],'tug_walk_seconds':sum(b-a for a,b in r['windows']),
            'tug_walk_max_horizontal_speed':r['max_horizontal_speed'],'total_observed_seconds':3+m['turn_observed_seconds'],
            'separate_walk_mean':float(base[i,0]),'same_tug_walk_mean':float(feature[0])})
    keep=np.array(keep);assert len(keep)==176 and sum(d['y'][keep]>=20)==8
    np.savez_compressed(HERE/'tug-speed-traces.npz',**traces)
    configs={}
    for cohort,indices,views in [('all237',np.arange(237),{'Current':base,'Inverse':inverse}),
                                 ('common176',keep,{'Current':base[keep],'Inverse':inverse[keep],'SameTUG':np.stack(tug)})]:
        out=HERE/cohort;out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'input.npz',ids=d['ids'][indices],y=d['y'][indices],**views)
        manifests=rows if cohort=='all237' else records
        save(cohort+'/manifest.json',manifests)
        mapping={int(j):k for k,j in enumerate(indices)};remap=lambda ix:[mapping[j] for j in ix if j in mapping]
        plans=[]
        for rep in json.loads((PRIOR/'splits.json').read_text()):
            plans.append({**rep,'splits':[{'fold':f['fold'],'train':remap(f['train']),'test':remap(f['test']),
                'inner':[{'train':remap(v['train']),'validation':remap(v['validation'])} for v in f['inner']]} for f in rep['splits']]})
        save(cohort+'/splits.json',plans)
        configs[cohort]={'people':len(indices),'long_people':int(sum(d['y'][indices]>=20)),'source_indices':indices.tolist(),'views':list(views),
            'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json']}}
    dependencies=[PRIOR/'input.npz',PRIOR/'manifest.json',PRIOR/'splits.json',PRIOR/'predictions.csv',PRIOR/'selections.json',EXT/'study.py',
        scout.TURN/'geometry.py',scout.TURN/'prepare.py',HERE.parent/'20261030-opencap415-tug-timeseries/prepare.py']
    protocol={'user_choice':'keep3seconds; all237 Current vs Inverse; common176 all3 models, all8 long retained',
        'cohorts':configs,'features':['mean horizontal speed OR reciprocal of mean','speed standard deviation','mean absolute adjacent speed change','same original turn duration'],
        'inverse':'replace only first feature by1/v; no reciprocal of noisy instantaneous speed, no target transformation, no clipping needed because all237 means>.05m/s; unconstrained Ridge to isolate feature transformation',
        'tug_walk_selection':'original file and turn boundaries; smoothed chair-axis progress>=max(0.3m,10percent excursion); legs adjacent to turn; aim1.5s each, reallocate shortage to other leg to total3s; no label-based window choice',
        'qc':'native horizontal pelvis speed<=6m/s; no reprocessing to rescue one failed record; no imputation or time stretching',
        'features_aggregation':'30Hz speed from native source; concatenate per-leg speed samples keeping91 total, within-leg changes only90 intervals; no jump across disconnected legs',
        'model':'same Ridge SVD, alpha0.1/1/10/100, train-only scaling and inner MAE choice; negative output floor0, raw saved, no maximum clipping',
        'split':'unchanged prior outer5partitions x5fold and inner3fold; common176 restrict memberships without reshuffling; sub283 absent everywhere',
        'observation':'3s nonturn walk plus unchanged D-second turn in each arm; two TUG legs may be discontinuous; full recording used for selection; not total acquisition time',
        'primary':'Inverse minus Current within all237; SameTUG minus Current within common176; other pair descriptive',
        'metrics':'person-equal MAE, underprediction=max(target-pred,0), overprediction=max(pred-target,0); >=20 descriptive8people; pairedperson bootstrap10000 seed20261031 conditional on fitted models; all5 partitions and eachperson',
        'limitations':['common176 selected slower population, not whole237; do not compare across cohort','same-visit label, exact timed trial cone/line uncertain','pre/post turn proximity represents approach/departure gait; automatic boundaries not manually validated','reused development population, no external validation'],
        'source_hashes':{str(p.relative_to(HERE.parent)):sha(p) for p in dependencies},'code_sha256':sha(HERE/'study.py'),'scout_sha256':sha(HERE/'scout.py'),'scout_results_sha256':sha(HERE/'scout.json')}
    save('protocol.json',protocol)
    save('exclusions.json',[r for r in audit if r['status']!='passed'])
    print(json.dumps({'cohorts':{k:{a:v[a] for a in ['people','long_people','views']} for k,v in configs.items()},'min_speed':float(base[:,0].min()),'max_inverse':float(inverse[:,0].max())}),flush=True)
if __name__=='__main__':prepare()
