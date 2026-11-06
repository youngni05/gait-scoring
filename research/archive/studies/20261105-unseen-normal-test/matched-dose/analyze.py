"""Usage: analyze.py interim | final. Interim is CNN bag only and issues no verdict."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tmp/ljmu-analysis-python-deps'))
import numpy as np
BIG=Path('D:/gait-research-data/D_P/unseen-normal-test-20261105');SRC=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104');MATCH=Path('D:/gait-research-data/D_P/matched-normal100-20261105')
stage=sys.argv[1];assert stage in('interim','final')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
meta=read(PARENT/'preparation.json');proto=read(HERE/'protocol.json');TOL=proto['validation']['replay_tolerance_points']
mem=read(ROOT/'studies/20261105-matched-normal100/membership.json');ens=read(ROOT/'studies/20261104-ljmu100-mixed/bundle/ensemble.json')
families=['CNN'] if stage=='interim' else ['CNN','STT','bilateral']
for arm in ['adult','child']:
    for fam in families:
        rec=read(HERE/f'infer-{arm}-{fam}.json');assert rec['replay_max_error']<=TOL,(arm,fam,rec)
        covered=set()
        for a in (ROOT/'studies/20261105-matched-normal100/audits'/arm).glob(fam+'-*.json'):
            v=read(a)
            if v.get('passed'):covered|={x['fold'] for x in v['folds']}
        assert covered==set(range(5)),(arm,fam,covered)
Pz={(a,f):np.load(BIG/f'dose-{a}-{f}.npz') for a in ['baseline','adult','child'] for f in families}
def score(arm,n,f):
    w=ens[f]['cnn_weight'];t=ens[f]['fraction'];g=lambda fam,k:Pz[arm,fam][f'{n}__{f}__{k}']
    if stage=='interim':return g('CNN','bag')
    b=w*g('CNN','bag')+(1-w)*g('STT','bag');bil=g('bilateral','refit');c=b.mean(1);d=(1-t)*(b[:,0]-b[:,1])+t*(bil[:,0]-bil[:,1])
    return np.stack([c+d/2,c-d/2],axis=1)
def ext(arm,n):return np.mean([score(arm,n,f) for f in range(5)],axis=0).mean(1)
def group(v,i):return np.array([v[i==q].mean() for q in sorted(set(i.tolist()))])
def ci(v):
    b=v[np.random.default_rng(20261105).integers(0,len(v),(10000,len(v)))].mean(1);return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]
livep=np.array(meta['live_people']);lp=np.array(meta['ljmu_person']);nums=np.array(meta['self_numbers']);norm=np.isin(nums,meta['self_normal']);abn=np.isin(nums,meta['self_abnormal'])
trained={r['person'] for r in mem['adult']};un=~np.isin(lp,list(trained));assert len(set(lp[un]))==55
def summary(arm):
    s=ext(arm,'self');o=dict(self_normal_mean=float(s[norm].mean()),self_normal_min=float(s[norm].min()),self_normal_max=float(s[norm].max()),self_abnormal_mean=float(s[abn].mean()),self_abnormal_min=float(s[abn].min()),self_abnormal_max=float(s[abn].max()),
        gap=float(s[norm].mean()-s[abn].mean()),ranges_overlap=bool(s[abn].max()>=s[norm].min()),videos={int(k):round(float(v),2) for k,v in zip(nums,s)})
    if arm!='child':
        l=group(ext(arm,'live'),livep);o.update(live_mean=float(l.mean()),live_sd=float(l.std(ddof=1)),live_ci95=ci(l))
    j=ext(arm,'ljmu_original')
    if arm!='child':
        q=group(j[un],lp[un]);o.update(ljmu55_mean=float(q.mean()),ljmu55_sd=float(q.std(ddof=1)),ljmu55_ci95=ci(q))
    if arm!='adult':
        q=group(j,lp);o.update(ljmu64_mean=float(q.mean()),ljmu64_sd=float(q.std(ddof=1)),ljmu64_ci95=ci(q))
    return o
R={'stage':stage,'model':'CNN_bag' if stage=='interim' else 'fusion','external':{a:summary(a) for a in ['baseline','adult','child']}}
# Stanford held-out
with np.load(MATCH/'child/data.npz') as z:y=z['y'][:1061];ids=z['person_id'][:1061];fold=z['fold'][:1061]
saved=np.load(SRC/'comparison-predictions.npz');s=saved['source']=='Stanford'
assert np.array_equal(saved['y'][s],y)
key='CNN_bag' if stage=='interim' else 'fusion';base=saved['baseline__'+key][s]
def held(arm):
    def load(fam,k):
        o=np.full_like(y,np.nan,dtype=float)
        for f in range(5):
            with np.load(MATCH/'received/data'/arm/'runs/mixed'/fam/f'fold{f}'/'predictions.npz') as z:o[z['row_index'][z['row_index']<1061]]=z[k][z['row_index']<1061]
        return o
    if stage=='interim':return load('CNN','bag')
    p=np.empty_like(y,dtype=float);c=load('CNN','bag');t_=load('STT','bag');b_=load('bilateral','refit')
    for f in range(5):
        m=fold==f;w=ens[f]['cnn_weight'];t=ens[f]['fraction'];b=w*c[m]+(1-w)*t_[m];co=b.mean(1);d=(1-t)*(b[:,0]-b[:,1])+t*(b_[m][:,0]-b_[m][:,1]);p[m]=np.stack([co+d/2,co-d/2],axis=1)
    return p
def pv(p,mask):
    u=np.unique(ids[mask.any(1)]);return np.array([np.abs((p-y)[mask&(ids==q)[:,None]]).mean() for q in u]),np.array([(p-y)[mask&(ids==q)[:,None]].mean() for q in u])
masks={'all':np.ones_like(y,bool),'ge90':y>=90,'ge100':y>=100,'lt70':y<70};R['stanford']={}
for name,m in masks.items():
    b,bb=pv(base,m);row=dict(people=len(b),baseline=dict(mae=float(b.mean()),bias=float(bb.mean())))
    for arm in ['adult','child']:
        e,bi=pv(held(arm),m);row[arm]=dict(mae=float(e.mean()),bias=float(bi.mean()),delta=float((e-b).mean()),delta_ci95=ci(e-b))
    R['stanford'][name]=row
Y=y.ravel();R['slope']={'baseline':float(np.polyfit(Y,base.ravel(),1)[0]),**{a:float(np.polyfit(Y,held(a).ravel(),1)[0]) for a in ['adult','child']}}
R['correlation']={'baseline':float(np.corrcoef(Y,base.ravel())[0,1]),**{a:float(np.corrcoef(Y,held(a).ravel())[0,1]) for a in ['adult','child']}}
if stage=='final':
    v={}
    for arm in ['adult','child']:
        e=R['external'][arm];st=R['stanford'];d1=(e['live_mean']>=90 and e['self_normal_mean']>=90) if arm=='adult' else (e['self_normal_mean']>=90 and e['ljmu64_mean']>=90)
        d2=e['gap']>=12.67 and not e['ranges_overlap'];d3=st['ge90'][arm]['delta_ci95'][1]<=.5;d4=st['all'][arm]['delta_ci95'][1]<=.1
        v[arm]=dict(D1=d1,D2=d2,D3=d3,D4=d4,overall='small_dose_sufficient' if d1 and d2 and d3 and d4 else 'not_all_passed')
    R['verdict']=v
(HERE/f'results-{stage}.json').write_text(json.dumps(R,indent=1,ensure_ascii=False),encoding='utf-8')
print(json.dumps(R,indent=1,ensure_ascii=False))
