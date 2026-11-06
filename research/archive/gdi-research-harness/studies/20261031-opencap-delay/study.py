from audit import *
import joblib
spec=importlib.util.spec_from_file_location('old_fit',HERE.parent/'20261031-opencap-extrapolation/study.py')
oldfit=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldfit)
ALPHAS=[.1,1.,10.,100.]
def fit(x,y,tr,te,alpha):return oldfit.fit(x,y,tr,te,'Ridge',alpha)

def features(a,windows,axis,threshold=.1):
    speeds=[];hs=[];axs=[];peaks=[];frozen=[]
    for w in windows:
        t,pos=grid(a,w);velocity=np.gradient(pos,1/30,axis=0)
        speeds.append(np.linalg.norm(velocity,axis=1).astype(np.float32))
        h,ax=motion(pos,axis);hs.extend(h[:-1]);axs.extend(ax[:-1])
        native=(a['time']>=w[0]-1e-7)&(a['time']<=w[1]+1e-7)
        pp=np.column_stack([a[k][native] for k in ['pelvis_tx','pelvis_tz']]);tt=a['time'][native]
        peaks.append(float(np.linalg.norm(np.gradient(pp,tt,axis=0),axis=1).max()))
        body=np.column_stack([a[k][native] for k in a.dtype.names if k!='time'])
        frozen.append(float(max([0]+[(r-l)*float(np.median(np.diff(tt))) for l,r in ranges(np.all(np.diff(body,axis=0)==0,axis=1))])))
    joined=np.concatenate([v[:-1] for v in speeds[:-1]]+[speeds[-1]])
    changes=np.concatenate([np.abs(np.diff(v)) for v in speeds])
    assert len(joined)==91 and len(changes)==90 and len(hs)==len(axs)==90
    speed=np.array([joined.mean(),joined.std(),changes.mean()],dtype=float)
    delay=np.array([np.sum(np.array(axs)<threshold)/30,np.sum(np.array(hs)<threshold)/30])
    return speed,delay,{'max_native_horizontal_speed':max(peaks),'max_identical_body_seconds':max(frozen)}

def prepare():
    assert not (HERE/'protocol.json').exists()
    old=dict(np.load(PREV/'common176/input.npz'));rows=json.loads((PREV/'common176/manifest.json').read_text())
    audit={r['id']:r for r in json.loads((HERE/'audit.json').read_text())}
    assert len(rows)==176 and sum(old['y']>=20)==8 and 'sub-283' not in old['ids']
    views={'Current':old['Current'].copy(),'SameTUG':old['SameTUG'].copy(),'ResampledSpeed':[],'Delay':[],'Delay05':[],'Delay15':[]};manifest=[]
    for i,m in enumerate(rows):
        r=audit[m['id']];assert r['status']=='passed';a=read(TURN/'raw'/m['turn_file'])
        windows=r['windows'];speed,delay,qc=features(a,windows,m['axis'])
        fallback=qc['max_native_horizontal_speed']>6 or qc['max_identical_body_seconds']>=.25
        if fallback:windows=m['tug_walk_windows'];speed,delay,_=features(a,windows,m['axis'])
        base=np.r_[speed,old['SameTUG'][i,3]]
        views['ResampledSpeed'].append(base);views['Delay'].append(np.r_[base,delay])
        for name,th in [('Delay05',.05),('Delay15',.15)]:views[name].append(np.r_[base,features(a,windows,m['axis'],th)[1]])
        manifest.append({**m,'candidate_window':r['candidate'],'proposed_windows':r['windows'],'observed_windows':windows,
            'candidate_qc':qc,'fallback_to_original':fallback,'delay_features_seconds':delay.tolist()})
    out=HERE/'common176';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'input.npz',ids=old['ids'],y=old['y'],**{k:np.array(v) for k,v in views.items()})
    save('common176/manifest.json',manifest);(out/'splits.json').write_bytes((PREV/'common176/splits.json').read_bytes())
    sources=[PREV/'common176'/n for n in ['input.npz','manifest.json','splits.json','predictions.csv']]+[TURN/'geometry.py',PREV/'scout.py',HERE.parent/'20261031-opencap-extrapolation/study.py']
    protocol={'cohorts':{'common176':{'people':176,'long_people':8,'views':list(views),'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json']}}},
      'primary':['Delay minus ResampledSpeed isolates extra delay descriptors on identical frames','Delay minus SameTUG measures combined window and feature change'],
      'comparison':'Current uses prior separate10m walking3s; SameTUG original within-TUG3s; ResampledSpeed uses same3s budget reallocated; Delay adds low axial and horizontal speed seconds in those same3s',
      'selection':'eligible non-turn legs from previous chair-distance mask; lowest mean smoothed absolute axial speed1.5s on0.1s search grid; remaining1.5s from original selected3s excluding overlap; no clinical target used',
      'observation':'exact3seconds nonturn plus identical original turn D per person in every model; full recording is scanned to select windows, so this is NOT a3second total acquisition protocol; no full-record delay length, absolute timestamps, total recording length or label-derived duration enters features',
      'delay_features':'two totals in selected3s: |axial velocity|<0.1m/s; horizontal speed<0.1m/s;7sample nearest-edge smoothing separately inside each30Hz window, no unobserved samples used; sensitivity0.05 and0.15, not used to select winner',
      'qc':'proposed windows native root speed>6m/s or all MOT coordinates exactly frozen>=0.25s => use original3s for all resampled arms; thresholds are flags, not proof of validity; TRC is retrospective quality review only',
      'model':'same train-only StandardScaler/Ridge SVD alpha.1/1/10/100, inner MAE, floor0/no uppercap',
      'split':'byte-identical previous176 outer5 partitions x5 folds and inner3fold; excludedsub283 everywhere',
      'metrics':'person-equal errors averaged over5 held-out predictions; all and target>=20 subgroup8; conditional pairedperson bootstrap10000 seed20261031; not external validation',
      'limitations':['176 common population retained; audit237 but cannot imply all237 training','same-visit clinical TUG; exact timed trial uncertain','no RGB in public archive; real stop versus tracking artifact cannot be proved from correlated pipeline outputs','window search sees fullrecord; only feature observation budget matched','chosen slow windows can include near-chair transitional movement; no confirmed clinical pause annotation','reused development population; only8long cases'],
      'source_hashes':{str(p.relative_to(HERE.parent)):sha(p) for p in sources},'code_sha256':sha(HERE/'study.py'),'audit_code_sha256':sha(HERE/'audit.py'),'audit_sha256':sha(HERE/'audit.json')}
    save('protocol.json',protocol)
    print(json.dumps({'people':176,'long':8,'views':list(views),'fallback':[m['id'] for m in manifest if m['fallback_to_original']],
      'long_features':[{k:m[k] for k in ['id','delay_features_seconds','candidate_qc']} for m in manifest if m['target']>=20]}),flush=True)
if __name__=='__main__':prepare()
