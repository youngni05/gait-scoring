"""Replay first, then preregistered Stanford-only calibration comparison."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tmp/ljmu-analysis-python-deps'))
import numpy as np
BIG=Path('D:/gait-research-data/D_P/unseen-normal-test-20261105');SRC=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proto=read(HERE/'protocol.json');TOL=proto['validation']['replay_tolerance_points']
meta=read(PARENT/'preparation.json');par=read(PARENT/'results.json');ens=read(ROOT/'studies/20261104-ljmu100-mixed/bundle/ensemble.json')
with np.load(SRC/'data.npz') as z:d={k:z[k] for k in z.files if k!='x'}
y=d['y'];ids=d['person_id'];stan=d['source']=='Stanford';fold=d['fold']
RAW={f:np.load(BIG/f'member-raw-{f}.npz') for f in ['CNN','STT','bilateral']}
def coef(p,t):return np.linalg.lstsq(np.column_stack([p.reshape(-1),np.ones(p.size)]),t.reshape(-1),rcond=None)[0]
members={};ols_err=0.
for fam in RAW:
    for f in range(5):
        root=SRC/'received/data/runs/mixed'/fam/f'fold{f}';done=read(root/'complete.json');rows=[]
        for m in done['members']:
            with np.load(root/m['tag']/'replay.npz') as z:tr=z['train_row'];tp=z['train_raw'];er=z['evaluation_row'];ep=z['evaluation_raw']
            assert not set(ids[tr])&set(ids[er]);keep=stan[tr]
            sc=coef(tp[keep],y[tr][keep]);xx=tp[keep].reshape(-1);yy=y[tr][keep].reshape(-1)
            sl=np.dot(xx-xx.mean(),yy-yy.mean())/np.dot(xx-xx.mean(),xx-xx.mean());ols_err=max(ols_err,float(np.abs(sc-[sl,yy.mean()-sl*xx.mean()]).max()))
            rows.append(dict(tag=m['tag'],excluded=m['excluded_from_bag'],mixed=np.array(m['calibration']),stan=sc,er=er,ep=ep))
        members[fam,f]=rows
assert ols_err<1e-8
def bag(fam,f,cal,raw_of):
    v=[raw_of(m)*m[cal][0]+m[cal][1] for m in members[fam,f] if not m['excluded']]
    return np.mean(v,axis=0)
def refit(fam,f,cal,raw_of):
    m=next(m for m in members[fam,f] if m['tag']=='refit');return raw_of(m)*m[cal][0]+m[cal][1]
def fusion(f,cal,raw_of):
    w=ens[f]['cnn_weight'];t=ens[f]['fraction'];b=w*bag('CNN',f,cal,raw_of)+(1-w)*bag('STT',f,cal,raw_of);bil=refit('bilateral',f,cal,raw_of)
    common=b.mean(1);diff=(1-t)*(b[:,0]-b[:,1])+t*(bil[:,0]-bil[:,1]);return np.stack([common+diff/2,common-diff/2],axis=1)
def stanford(cal):
    p=np.full_like(y,np.nan,dtype=float)
    for f in range(5):p[members['CNN',f][0]['er']]=fusion(f,cal,lambda m:m['ep'])
    return p
def external(n,cal,f):
    fam={id(v):k for k,vs in members.items() for v in vs}
    return fusion(f,cal,lambda m:RAW[fam[id(m)][0]][f"{n}__{f}__{m['tag']}"])
saved=np.load(SRC/'comparison-predictions.npz')
P={'baseline':saved['baseline__fusion'],'mixed':stanford('mixed'),'mixed_stanford_cal':stanford('stan')}
checks={'stanford_mixed':float(np.abs(P['mixed'][stan]-saved['mixed__fusion'][stan]).max())}
ljfold=np.array(meta['ljmu_fold']);lp=np.array(meta['ljmu_person']);livep=np.array(meta['live_people'])
def lj(cal):
    o=np.full((881,2),np.nan)
    for f in range(5):o[ljfold==f]=external('ljmu_original',cal,f)
    return o
checks['ljmu_mixed']=float(np.abs(lj('mixed')-saved['mixed__fusion'][~stan]).max())
def group(v,i):return np.array([v[i==q].mean() for q in sorted(set(i.tolist()))])
ext={c:{n:np.mean([external(n,c,f) for f in range(5)],axis=0).mean(1) for n in ['live','self']} for c in ['mixed','stan']}
nums=np.array(meta['self_numbers']);norm=np.isin(nums,meta['self_normal']);abn=np.isin(nums,meta['self_abnormal'])
checks['live_mixed']=abs(float(group(ext['mixed']['live'],livep).mean())-par['live']['mixed']['person_mean'])
checks['self_mixed']=abs(float(ext['mixed']['self'][norm].mean())-par['self']['mixed']['normal_mean'])
print(json.dumps(checks,indent=1));assert max(checks.values())<=TOL,'REPLAY_FAILED'
def person_values(p,mask):
    people=np.unique(ids[mask.any(axis=1)]);return people,np.array([np.abs((p-y)[mask&(ids==q)[:,None]]).mean() for q in people]),np.array([(p-y)[mask&(ids==q)[:,None]].mean() for q in people])
masks={'all':stan[:,None]&np.ones_like(y,bool),'ge90':stan[:,None]&(y>=90),'ge100':stan[:,None]&(y>=100),'lt70':stan[:,None]&(y<70)}
R={'replay':checks,'ols_closed_form_max_error':ols_err,'stanford':{}}
for name,m in masks.items():
    row={};_,b,_b=person_values(P['baseline'],m);n=len(b);draw=np.random.default_rng(20261105).integers(0,n,(10000,n))
    for arm in P:
        _,e,bi=person_values(P[arm],m);de=e-b
        row[arm]=dict(mae=float(e.mean()),bias=float(bi.mean()),delta_vs_baseline=float(de.mean()),delta_ci95=[float(v) for v in np.quantile(de[draw].mean(1),[.025,.975])])
    row['people']=n;row['side_observations']=int(m.sum());R['stanford'][name]=row
Y=y[stan].ravel();R['slope']={a:float(np.polyfit(Y,P[a][stan].ravel(),1)[0]) for a in P};R['correlation']={a:float(np.corrcoef(Y,P[a][stan].ravel())[0,1]) for a in P}
R['prediction_sd']={a:float(P[a][stan].std()) for a in P}
ranges={}
def ext_row(s,l):
    n=s[norm];a=s[abn];p=group(l,livep);dr=np.random.default_rng(20261105).integers(0,len(p),(10000,len(p)))
    return dict(live_mean=float(p.mean()),live_sd=float(p.std(ddof=1)),live_ci95=[float(v) for v in np.quantile(p[dr].mean(1),[.025,.975])],self_normal_mean=float(n.mean()),self_normal_min=float(n.min()),self_normal_max=float(n.max()),
        self_abnormal_mean=float(a.mean()),self_abnormal_min=float(a.min()),self_abnormal_max=float(a.max()),gap=float(n.mean()-a.mean()),ranges_overlap=bool(a.max()>=n.min()),videos={int(k):round(float(v),2) for k,v in zip(nums,s)})
R['external']={'mixed':ext_row(ext['mixed']['self'],ext['mixed']['live']),'mixed_stanford_cal':ext_row(ext['stan']['self'],ext['stan']['live'])}
R['ljmu_mean']={'mixed':float(group(lj('mixed').mean(1),lp).mean()),'mixed_stanford_cal':float(group(lj('stan').mean(1),lp).mean())}
ml=[(k,m['mixed'],m['stan']) for k,v in members.items() for m in v if m['tag']=='refit']
R['refit_calibration']=[dict(family=k[0],fold=k[1],mixed_slope=float(a[0]),mixed_intercept=float(a[1]),stanford_slope=float(b[0]),stanford_intercept=float(b[1])) for k,a,b in ml]
e=R['external']['mixed_stanford_cal'];up=R['stanford']['ge90']['mixed_stanford_cal']['delta_ci95'][1]
c1=up<=.5;c2=e['live_mean']>=90 and e['self_normal_mean']>=90;c3=e['gap']>=12.67 and not e['ranges_overlap']
np.savez_compressed(BIG/'stanford-cal-fusion.npz',pred=P['mixed_stanford_cal']);
R['verdict']=dict(C1_stanford_high_recovery=c1,C1_upper=up,C2_external_kept=c2,C3_discrimination_kept=c3,overall='both_effects_coexist' if c1 and c2 and c3 else 'not_all_passed')
(HERE/'results.json').write_text(json.dumps(R,indent=1,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in R.items() if k!='refit_calibration'},indent=1,ensure_ascii=False))
