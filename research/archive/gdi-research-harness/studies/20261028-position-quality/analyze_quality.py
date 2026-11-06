import csv, hashlib, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent; H=HERE.parents[1]; ROOT=H.parent
OLD=H/'studies/20261027-window-ablation'
PREV=H/'control/desktop-sessions/20261027T153603Z-79100a'
SHORT=H/'control/desktop-sessions/20261026T130506Z-e876e7'
DATA=ROOT/'outputs/gdi-ridge-experiment-20261016/windows'
OFFSETS={'front':0,'middle':59,'last':119}

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def rank(x):
    ordered=sorted(set(x));return np.array([sum(v<t for v in x)+(sum(v==t for v in x)+1)/2 for t in x])
def rho(x,y):
    x,y=rank(x),rank(y)
    return None if np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1])
def width(x):return float(np.percentile(x,95)-np.percentile(x,5))

def measures(q,scale):
    xy=q[:,:,:2].astype(float); c=q[:,:,2]; low=xy[:,1:7]-xy[:,0:1]
    lengths=np.stack([np.linalg.norm(xy[:,a]-xy[:,b],axis=1) for a,b in ((1,2),(2,3),(4,5),(5,6))],axis=1)
    def knee(a,b,c):
        u=xy[:,a]-xy[:,b];v=xy[:,c]-xy[:,b];den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        good=den>1e-12
        return np.degrees(np.arccos(np.clip((u[good]*v[good]).sum(axis=1)/den[good],-1,1)))
    torso=xy[:,8]-xy[:,0]
    good=(q[:,11,2]>0)&(q[:,14,2]>0)&(np.linalg.norm(torso,axis=1)>1e-12)
    assert good.sum()>0
    widths=np.linalg.norm(xy[good,11]-xy[good,14],axis=1)/np.linalg.norm(torso[good],axis=1)
    trunk=np.degrees(np.arctan2(torso[good,0],-torso[good,1]))
    return {
      'lower_confidence_mean':float(c[:,1:7].mean()),
      'lower_below_0_3_fraction':float((c[:,1:7]<.3).mean()),
      'all_joint_missing_fraction':float((c==0).mean()),
      'projected_bone_length_cv':float((lengths.std(axis=0)/lengths.mean(axis=0)).mean()),
      'relative_lower_acceleration_p90':float(np.percentile(np.linalg.norm(np.diff(low,n=2,axis=0),axis=2),90)/scale),
      'ankle_horizontal_excursion':float(np.mean([width((xy[:,a,0]-xy[:,0,0])/scale) for a in (3,6)])),
      'knee_angle_range_deg':float(np.mean([width(knee(1,2,3)),width(knee(4,5,6))])),
      'trunk_angle_range_deg':width(trunk),
      'shoulder_width_over_torso':float(np.median(widths)),
      'pelvis_horizontal_displacement':float((xy[-1,0,0]-xy[0,0,0])/scale),
      'valid_shoulder_torso_frames':int(good.sum()),
      'lower_missing_fraction':float((c[:,1:7]==0).mean())}

def main():
    assert (HERE/'plan.md').exists() and not (HERE/'result.json').exists()
    sealed={a['path']:a['sha256'] for a in load(OLD/'study_receipt.json')['artifacts']}
    tracked={}
    for name in ('raw_features.npz','new_validation_models.npz','sources.json','comparison.json'):
        p=OLD/name;assert sha(p)==sealed[p.relative_to(H).as_posix()];tracked[p]=sha(p)
    prev_seal={a['path']:a['sha256'] for a in load(PREV/'receipt.json')['artifacts']}
    p=PREV/'predictions.csv';assert sha(p)==prev_seal[p.relative_to(H).as_posix()];tracked[p]=sha(p)
    model=SHORT/'length_models.npz';seals={a['path']:a['sha256'] for a in load(SHORT/'receipt.json')['artifacts']}
    assert sha(model)==seals[model.relative_to(H).as_posix()];tracked[model]=sha(model)
    with np.load(OLD/'raw_features.npz',allow_pickle=False) as f:
        people,videos,split=(f[k].copy() for k in ('patient_ids','video_ids','split'))
        features={m:f[m+'_local'].copy() for m in OFFSETS}
    assert len(people)==243 and len(set(people))==212
    assert len(set(people[split=='train']))==175 and len(set(people[split=='validation']))==37
    assert not set(people[split=='train'])&set(people[split=='validation'])
    excluded=load(H/'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json')
    assert not set(people)&set(excluded['original_test_patient_ids'])
    source=load(OLD/'sources.json')['sha256']; rows=[]
    for person,video,role in zip(people,videos,split):
        path=DATA/(str(video)+'.npz');assert sha(path)==source[path.relative_to(ROOT).as_posix()];tracked[path]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            q=z['input'].copy();ids=z['frame_ids']
        assert q.shape==(243,17,3) and np.isfinite(q).all() and (np.diff(ids)==1).all()
        assert (q[:,1:7,2]>0).all()
        length=np.stack([np.linalg.norm(q[:,a,:2]-q[:,b,:2],axis=1) for a,b in ((1,2),(2,3),(4,5),(5,6))])
        scale=float(np.median(length));assert scale>1e-8
        for pos,start in OFFSETS.items():
            m=measures(q[start:start+124],scale)
            assert all(np.isfinite(v) for v in m.values())
            rows.append(dict(patient_id=str(person),video_id=str(video),split=str(role),position=pos,**m))
    metrics=[k for k in rows[0] if k not in ('patient_id','video_id','split','position','valid_shoulder_torso_frames')]
    with (HERE/'quality_rows.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    by=defaultdict(list)
    for r in rows:by[(r['split'],r['position'],r['patient_id'])].append(r)
    def values(role,pos,k):return {p:float(np.mean([r[k] for r in rr])) for (s,m,p),rr in by.items() if (s,m)==(role,pos)}
    summaries={s:{m:{k:float(np.mean(list(values(s,m,k).values()))) for k in metrics} for m in OFFSETS} for s in ('train','validation')}
    with (PREV/'predictions.csv').open(encoding='utf-8') as f:pred_rows=list(csv.DictReader(f))
    pred={(r['patient_id'],r['video_id'],r['method']):[float(r['prediction_'+s]) for s in ('L','R')] for r in pred_rows}
    true={r['video_id']:[float(r['actual_'+s]) for s in ('L','R')] for r in pred_rows}
    val=split=='validation';pv=people[val];vv=videos[val]
    associations={}
    for pos in ('middle','last'):
        dp=defaultdict(list);ap=defaultdict(list)
        for person,video in zip(pv,vv):
            delta=np.array(pred[(str(person),str(video),pos)])-np.array(pred[(str(person),str(video),'front')])
            dp[str(person)].append(float(delta.mean()));ap[str(person)].append(float(np.abs(delta).mean()))
        keys=sorted(dp);d=np.array([np.mean(dp[k]) for k in keys]);a=np.array([np.mean(ap[k]) for k in keys])
        correlations={}
        for k in metrics:
            before,after=values('validation','front',k),values('validation',pos,k)
            change=np.array([after[p]-before[p] for p in keys])
            correlations[k]=dict(mean_change=float(change.mean()),signed_spearman=rho(change,d),absolute_spearman=rho(abs(change),a))
        associations[pos]=dict(people=len(keys),metrics=correlations)
    heads={}
    with np.load(model,allow_pickle=False) as z:
        heads['front']={k:z['short124__'+k].copy() for k in ('scaler_mean','scaler_scale','coef','intercept')}
    with np.load(OLD/'new_validation_models.npz',allow_pickle=False) as z:
        for m in ('middle','last'):heads[m]={k:z[m+'_local__'+k].copy() for k in ('scaler_mean','scaler_scale','coef','intercept')}
    truth=np.array([true[str(v)] for v in vv]);matrix={};cross=[]
    for trained,p in heads.items():
        matrix[trained]={}
        for pos,x in features.items():
            output=((x[val]-p['scaler_mean'])/p['scaler_scale'])@p['coef'].T+p['intercept']
            errors=np.abs(output-truth).mean(axis=1)
            matrix[trained][pos]=float(np.mean([errors[pv==person].mean() for person in sorted(set(pv))]))
            for i,v in enumerate(vv):cross.append(dict(patient_id=str(pv[i]),video_id=str(v),trained_position=trained,input_position=pos,actual_L=truth[i,0],actual_R=truth[i,1],prediction_L=output[i,0],prediction_R=output[i,1]))
    reference=load(OLD/'comparison.json')['validation37']
    for m in OFFSETS:assert abs(matrix[m][m]-reference[m+'_local']['person_mae'])<1e-9
    with (HERE/'cross_heads.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cross[0]));w.writeheader();w.writerows(cross)
    assert all(sha(p)==d for p,d in tracked.items())
    save('sources.json',{p.relative_to(ROOT).as_posix():d for p,d in tracked.items()})
    result=dict(quality_people=212,quality_records=243,main_association_people=37,main_association_records=40,
        summary=summaries,associations=associations,cross_head_person_mae=matrix,
        minimum_valid_shoulder_torso_frames=min(r['valid_shoulder_torso_frames'] for r in rows),
        sources_unchanged=True,new_training=0,heldout38_or223_used=False,causal_effect_identified=False,
        limitations=['confidence not ground-truth coordinate error','2D changes confounded by view, camera motion and true movement','preselected fully detected lower-limb cohort','exploratory multiple associations on reused validation37','no new ordinal target or clinical threshold'])
    save('result.json',result)
    print(json.dumps(dict(validation_summary=summaries['validation'],last_associations=associations['last'],cross_head_person_mae=matrix),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
