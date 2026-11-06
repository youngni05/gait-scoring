from scout import *
from scipy.ndimage import uniform_filter1d,maximum_filter1d
import joblib
spec=importlib.util.spec_from_file_location('old_fit',HERE.parent/'20261031-opencap-extrapolation/study.py');oldfit=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldfit)
ALPHAS=[.1,1.,10.,100.]
def fit(x,y,tr,te,alpha):return oldfit.fit(x,y,tr,te,'Ridge',alpha)
def runs(mask):
    z=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(z==1),np.flatnonzero(z==-1)))
def episodes(slow,valid,t):
    mask=slow&valid
    for l,r in runs(~mask):
        if l>0 and r<len(mask) and r-l<=6 and valid[l:r].all():mask[l:r]=True
    for l,r in runs(mask):
        if r-l<15:mask[l:r]=False
    spans=[{'start':float(t[l]),'end':float(t[r]),'seconds':(r-l)/30} for l,r in runs(mask)]
    return [float(mask.sum()/30),max([0.]+[x['seconds'] for x in spans])],spans

def extract(a,m,s,budget):
    # Contains the original turn and immediately adjacent gait; expands backward first.
    original_end=max(m['turn_end'],max(w[1] for w in m['tug_walk_windows']))
    left=max(s['start'],original_end-budget);right=min(float(a['time'][-1]),left+budget)
    n=int(np.floor((right-left)*30+1e-5));t=left+np.arange(n+1)/30;right=float(t[-1])
    pos=np.column_stack([np.interp(t,a['time'],a[k]) for k in ['pelvis_tx','pelvis_tz']])
    rawv=np.gradient(pos,1/30,axis=0);speed=np.linalg.norm(rawv,axis=1)
    smooth=uniform_filter1d(pos,7,axis=0,mode='nearest');v=np.gradient(smooth,1/30,axis=0);horizontal=np.linalg.norm(v,axis=1);axial=abs(v@m['axis'])
    midpoint=(t[:-1]+t[1:])/2;eligible=np.zeros(n,bool)
    for leg in [s['pre'],s['post']]:
        if leg:eligible|=(midpoint>=leg[0]-1e-6)&(midpoint<=leg[1]+1e-6)
    initial=float(np.median(a['pelvis_ty'][a['time']<=a['time'][0]+.5]))
    height=np.interp(midpoint,a['time'],a['pelvis_ty']);knee=np.interp(midpoint,a['time'],(a['knee_angle_l']+a['knee_angle_r'])/2);hip=np.interp(midpoint,a['time'],(a['hip_flexion_l']+a['hip_flexion_r'])/2)
    seated=(height<initial+.12)&(knee>60)&(hip>30)
    # Native tracking jumps contaminate nearby interpolation/smoothing; mask them, never call them pauses.
    pp=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);nt=a['time'];jumps=np.flatnonzero(np.linalg.norm(np.diff(pp,axis=0),axis=1)/np.diff(nt)>6)
    bad=np.zeros(n,bool)
    for j in jumps:bad|=(midpoint>=nt[j]-.15)&(midpoint<=nt[j+1]+.15)
    valid=eligible&~seated&~bad
    assert valid.sum()>=15
    vals=speed[:-1][valid];adj=valid[:-1]&valid[1:]
    base=np.array([vals.mean(),vals.std(),abs(np.diff(speed[:-1]))[adj].mean(),m['turn_observed_seconds']],float)
    assert np.isfinite(base).all()
    delays={};events={}
    for name,th in [('10',.10),('15',.15)]:
        x,xe=episodes(axial[:-1]<th,valid,t);h,he=episodes(horizontal[:-1]<th,valid,t)
        delays[name]=np.r_[x,h];events[name]={'axial':xe,'horizontal':he}
    record={'window':[left,right],'observed_seconds':n/30,'valid_nonturn_seconds':float(valid.sum()/30),'seated_removed_seconds':float(sum(eligible&seated)/30),'tracking_removed_seconds':float(sum(eligible&~seated&bad)/30),
      'turn_seconds':m['turn_observed_seconds'],'delay_features':{k:v.tolist() for k,v in delays.items()},'events':events}
    traces={'time':t,'speed':speed,'horizontal':horizontal,'axial':axial,'valid':valid,'eligible':eligible,'seated':seated,'bad':bad}
    return base,delays,record,traces

def prepare():
    assert not (HERE/'protocol.json').exists();rows=json.loads((PREV/'common176/manifest.json').read_text());scouts=json.loads((HERE/'scout.json').read_text());olddata=dict(np.load(PREV/'common176/input.npz'))
    views={'Current':olddata['Current'].copy(),'SameTUG':olddata['SameTUG'].copy()}
    for b in [10,20,30]:
        for prefix in ['Speed','Delay10','Delay15']:views[f'{prefix}_{b}']=[]
    manifests=[];traces={}
    for i,(m,s) in enumerate(zip(rows,scouts)):
        assert m['id']==s['id']==olddata['ids'][i];a=read(TURN/'raw'/m['turn_file']);assert sha(TURN/'raw'/m['turn_file'])==m['turn_sha256'];windows={}
        for b in [10,20,30]:
            base,delay,r,tt=extract(a,m,s,b);views[f'Speed_{b}'].append(base)
            for suffix in ['10','15']:views[f'Delay{suffix}_{b}'].append(np.r_[base,delay[suffix]])
            windows[str(b)]=r
            for k,v in tt.items():traces[f'{m["id"]}_{b}_{k}']=v
        manifests.append({**m,'source_sample':i,'budgets':windows})
    views={k:np.asarray(v) for k,v in views.items()};np.savez_compressed(HERE/'traces.npz',**traces)
    cohorts={};exact=np.array([i for i,m in enumerate(manifests) if abs(m['budgets']['10']['observed_seconds']-10)<1e-6])
    for name,ix in [('capped176',np.arange(176)),('exact10',exact)]:
        out=HERE/name;out.mkdir(exist_ok=True);np.savez_compressed(out/'input.npz',ids=olddata['ids'][ix],y=olddata['y'][ix],**{k:v[ix] for k,v in views.items()});save(name+'/manifest.json',[manifests[i] for i in ix])
        mapping={int(j):k for k,j in enumerate(ix)};remap=lambda xs:[mapping[j] for j in xs if j in mapping]
        plans=[]
        for rep in json.loads((PREV/'common176/splits.json').read_text()):
            plans.append({**rep,'splits':[{'fold':f['fold'],'train':remap(f['train']),'test':remap(f['test']),'inner':[{'train':remap(v['train']),'validation':remap(v['validation'])} for v in f['inner']]} for f in rep['splits']]})
        save(name+'/splits.json',plans);cohorts[name]={'people':len(ix),'long_people':int(sum(olddata['y'][ix]>=20)),'views':list(views),'source_indices':ix.tolist(),'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json']}}
    source=[PREV/'common176'/n for n in ['input.npz','manifest.json','splits.json','predictions.csv']]+[PREV/'scout.py',HERE.parent/'20261031-opencap-extrapolation/study.py']
    protocol={'cohorts':cohorts,'budgets':[10,20,30],
      'window':'contains previous full turn+adjacent gait, expands backward first to first away-from-chair, then forward until budget or source end; nested windows; no padding or stretching; no clinical label used',
      'scope':'maximum budget, NOT everyone observed exactly10/20/30; exact10 sensitivity cohort with original memberships restricted; no exact20/30 cohort inference due inadequate people',
      'features':'same4 speed mean/std/adjacent absolute change + full unchanged turn D; each budget speed computed within observed eligible nonturn not-seated not-jump intervals; Delay adds axial total/max episode seconds and horizontal total/max episode seconds',
      'delay':'main0.10m/s sensitivity0.15;7point30Hz position smoothing within observed window; gaps<=0.20s bridged only if valid, episodes>=0.50s; max and sum never cross missing/chair/turn; full-record unobserved pause length never used',
      'qc':'exclude pointwise seated-like intervals height<initial0.5s median+0.12 AND bilateral mean knee>60 AND hip>30; native root jump>6m/s plus0.15s margin masked; no target-based exclusions',
      'model':'same Ridge SVD alpha0.1/1/10/100 chosen by innerMAE; train-only scaling; negativefloor0; no uppercap',
      'split':'original5partitionsx5outerfoldx3innerfold; sub283 absent from every stage; exact10 restrict people without re-shuffling',
      'primary':'Delay10_10/20/30 minus Speed at same budget, and each minus SameTUG; larger window vs Speed10; threshold0.15 sensitivity all reported',
      'metrics':'person-equal MAE/under/over/bias over5heldout predictions; >=20 subgroup8; paired10000person bootstrap conditional on fits seed20261031',
      'limits':['label sameperson/visit but timed trial uncertain','full-record phase/axis selection not prospective online','expanded views can cover almost full gait-turn performance, closer to measuring its duration than predicting from short gait; clinical totalTUG and totalrecord length are not features','no independent RGB; marker QC prior9people only','10/20/30 capped budgets do not represent fixed duration acquisition','reused development cohort and8long only'],
      'code_sha256':sha(HERE/'study.py'),'source_hashes':{str(p.relative_to(HERE.parent)):sha(p) for p in source},'scout_sha256':sha(HERE/'scout.json')}
    save('protocol.json',protocol)
    stats={str(b):{'full_budget':sum(abs(m['budgets'][str(b)]['observed_seconds']-b)<1e-6 for m in manifests),'observed_min_med_max':np.quantile([m['budgets'][str(b)]['observed_seconds'] for m in manifests],[0,.5,1]).tolist(),'tracking_mask_people':sum(m['budgets'][str(b)]['tracking_removed_seconds']>0 for m in manifests)} for b in [10,20,30]}
    save('coverage.json',stats);print(json.dumps({'cohorts':{k:{'people':v['people'],'long':v['long_people']} for k,v in cohorts.items()},'coverage':stats,'long':[{'id':m['id'],'budgets':{b:{k:r[k] for k in ['observed_seconds','valid_nonturn_seconds','delay_features','tracking_removed_seconds']} for b,r in m['budgets'].items()}} for m in manifests if m['target']>=20]}),flush=True)
if __name__=='__main__':prepare()
