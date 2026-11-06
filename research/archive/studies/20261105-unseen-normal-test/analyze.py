"""Replay checks first, then preregistered summaries. Stops if replay fails."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tmp/ljmu-analysis-python-deps'))
import numpy as np
BIG=Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta=read(HERE/'preparation.json');proto=read(HERE/'protocol.json');assert sha(HERE/'protocol.json')==meta['protocol_sha256']
TOL=proto['validation']['replay_tolerance_points']
P={f:np.load(BIG/f'pred-{f}.npz') for f in ['CNN','STT','bilateral']}
ens=read(ROOT/'studies/20261104-ljmu100-mixed/bundle/ensemble.json')
def fuse(n,arm,f):
    v=ens[f];w=v['cnn_weight'];t=v['fraction']
    cnn=P['CNN'][f'{n}__{arm}__{f}__bag'];stt=P['STT'][f'{n}__{arm}__{f}__bag'];bil=P['bilateral'][f'{n}__{arm}__{f}__refit']
    b=w*cnn+(1-w)*stt;common=b.mean(axis=1);diff=(1-t)*(b[:,0]-b[:,1])+t*(bil[:,0]-bil[:,1])
    return np.stack([common+diff/2,common-diff/2],axis=1)
fold=np.array(meta['ljmu_fold']);lperson=np.array(meta['ljmu_person'])
def heldout(n,arm,model='fusion'):
    out=np.full((881,2),np.nan)
    for f in range(5):
        p=fuse(n,arm,f) if model=='fusion' else P[model][f'{n}__{arm}__{f}__bag']
        out[fold==f]=p
    assert np.isfinite(out).all();return out
def unseen(n,arm,model='fusion'):
    each=np.stack([fuse(n,arm,f) if model=='fusion' else P[model][f'{n}__{arm}__{f}__bag'] for f in range(5)])
    return each
# Replay
checks={}
saved=np.load('D:/gait-research-data/D_P/ljmu100-mixed-20261104/comparison-predictions.npz')
lj=saved['source']=='LJMU'
for arm in ['baseline','mixed']:
    checks['ljmu_fusion_'+arm]=float(np.abs(heldout('ljmu_original',arm)-saved[arm+'__fusion'][lj]).max())
w=read(ROOT/'gdi-research-harness/studies/20261101-self-video-test01/windows.json');nums=np.array([v['number'] for v in w])
old=np.load('D:/gait-research-data/D_P/self-video-test01-20261101/predictions.npz')['best']
oldrec=np.stack([old[nums==n].mean(0) for n in meta['self_numbers']])
# The saved value averages window-level fusion; fusion is linear in window scores, so record means must match.
checks['self_fusion_baseline']=float(np.abs(unseen('self','baseline').mean(0)-oldrec).max())
print(json.dumps(checks,indent=1))
assert max(checks.values())<=TOL,'REPLAY_FAILED'
def group(v,ids):
    u=np.array(sorted(set(ids.tolist())));return np.array([v[ids==q].mean() for q in u])
def ci(v,seed=20261105):
    r=np.random.default_rng(seed);b=v[r.integers(0,len(v),(10000,len(v)))].mean(1);return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]
R={'replay':checks,'replay_tolerance':TOL}
# Live
lp=np.array(meta['live_people']);live={}
for arm in ['baseline','mixed']:
    each=unseen('live',arm);score=each.mean(0).mean(1);people=group(score,lp)
    live[arm]=dict(person_mean=float(people.mean()),person_sd=float(people.std(ddof=1)),person_min=float(people.min()),person_max=float(people.max()),ci95=ci(people),
        fold_group_means=[float(group(e.mean(1),lp).mean()) for e in each],records=score.round(2).tolist())
    for m in ['CNN','STT']:live[arm][m+'_bag']=float(group(unseen('live',arm,m).mean(0).mean(1),lp).mean())
R['live']=live
# Self
nums=np.array(meta['self_numbers']);norm=np.isin(nums,meta['self_normal']);abn=np.isin(nums,meta['self_abnormal']);self_={}
for arm in ['baseline','mixed']:
    each=unseen('self',arm);score=each.mean(0).mean(1)
    self_[arm]=dict(normal_mean=float(score[norm].mean()),normal_min=float(score[norm].min()),normal_max=float(score[norm].max()),
        abnormal_mean=float(score[abn].mean()),abnormal_min=float(score[abn].min()),abnormal_max=float(score[abn].max()),
        gap=float(score[norm].mean()-score[abn].mean()),ranges_overlap=bool(score[abn].max()>=score[norm].min()),
        videos={int(n):round(float(s),2) for n,s in zip(nums,score)},
        fold_normal_means=[float(e.mean(1)[norm].mean()) for e in each],fold_abnormal_means=[float(e.mean(1)[abn].mean()) for e in each])
    for m in ['CNN','STT']:
        s=unseen('self',arm,m).mean(0).mean(1);self_[arm][m+'_bag']=dict(normal=float(s[norm].mean()),abnormal=float(s[abn].mean()))
R['self']=self_
# Perturbations
conds=['ljmu_original','ljmu_frame_shuffle','ljmu_static_mean','ljmu_left_leg_stiff','ljmu_both_amp50'];pert={}
for arm in ['baseline','mixed']:
    base=group(heldout('ljmu_original',arm).mean(1),lperson);pert[arm]={}
    for c in conds:
        h=heldout(c,arm);s=group(h.mean(1),lperson);d=s-base
        pert[arm][c]=dict(person_mean=float(s.mean()),person_sd=float(s.std(ddof=1)),person_min=float(s.min()),person_max=float(s.max()),delta_from_original=float(d.mean()),delta_ci95=ci(d),
            left=float(group(h[:,0],lperson).mean()),right=float(group(h[:,1],lperson).mean()),
            person_share_ge95=float((s>=95).mean()),person_share_ge90=float((s>=90).mean()))
R['perturbations']=pert
# Preregistered verdicts
a=live['mixed']['person_mean'];b=self_['mixed']['normal_mean']
R['verdict']=dict(live_mixed_mean=a,self_normal_mixed_mean=b,overall='supports_normal_gait' if a>=90 and b>=90 else 'supports_source_recognition' if a<90 and b<90 else 'inconclusive',
    perturbation={c:('appearance_dependent' if pert['mixed'][c]['person_mean']>=95 else 'motion_responsive' if pert['mixed'][c]['delta_from_original']<=-10 else 'partial') for c in conds[1:]})
(HERE/'results.json').write_text(json.dumps(R,indent=1,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:R[k] for k in ['live','self','perturbations','verdict']},indent=1,ensure_ascii=False))
