import json,csv,hashlib,math,statistics
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;OLD=H/'studies/20261028-window-cause-controls';BASE=H/'studies/20261028-missing-augmentation-learning'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-8):assert abs(float(a)-float(b))<tol,(a,b)
def independent_detection(signal):
    n=len(signal);padded=[signal[0]]*2+list(signal)+[signal[-1]]*2;sm=[math.fsum(padded[i+j]*w for j,w in enumerate([1/9,2/9,3/9,2/9,1/9])) for i in range(n)];tt=[i-(n-1)/2 for i in range(n)];mean=statistics.mean(sm);slope=math.fsum(t*v for t,v in zip(tt,sm))/math.fsum(t*t for t in tt);x=[v-mean-slope*t for v,t in zip(sm,tt)];sd=statistics.pstdev(x)
    if sd<1e-8:return None,[],[]
    ac=[None]
    for lag in range(1,92):
        aa=x[:-lag];bb=x[lag:];ma=statistics.mean(aa);mb=statistics.mean(bb);a=[v-ma for v in aa];b=[v-mb for v in bb];den=math.sqrt(math.fsum(v*v for v in a)*math.fsum(v*v for v in b));ac.append(math.fsum(u*v for u,v in zip(a,b))/den if den>1e-16 else 0.)
    poss=[k for k in range(16,91) if ac[k]>=.35 and ac[k]>ac[k-1] and ac[k]>=ac[k+1]]
    if not poss:return None,[],[]
    P=poss[0];radius=P//2;sep=math.ceil(P/2);poss=[]
    for t in range(1,n-1):
        if x[t]>x[t-1] and x[t]>=x[t+1] and x[t]-max(min(x[max(0,t-radius):t+1]),min(x[t:min(n,t+radius+1)]))>=sd*.25:poss.append(t)
    peaks=[]
    for t in sorted(poss,key=lambda i:(-x[i],i)):
        if all(abs(t-u)>=sep for u in peaks):peaks.append(t)
    peaks.sort();cycles=[[a,b] for a,b in zip(peaks,peaks[1:]) if .65*P<=b-a<=1.35*P];return P,peaks,cycles

def independent_input(raw,times):
    arr=np.zeros((124,17,3),float)
    for i,t in enumerate(times):
        closest=round(float(t))
        if abs(t-closest)<1e-10:arr[i]=raw[closest];continue
        a=math.floor(float(t));b=a+1;w=t-a
        for j in range(17):
            if raw[a,j,2]>0 and raw[b,j,2]>0:
                arr[i,j,:2]=[(1-w)*float(raw[a,j,k])+w*float(raw[b,j,k]) for k in range(2)];arr[i,j,2]=min(raw[a,j,2],raw[b,j,2])
    valid=arr[:,:,:2][arr[:,:,2]>0];lo=valid.min(0);hi=valid.max(0);arr[:,:,:2]=(arr[:,:,:2]-(lo+hi)/2)*2/max(hi-lo);return np.clip(arr,-1,1).astype(np.float32)
def main():
    sources={**load(S/'sources.json'),**load(S/'inference_sources.json'),**load(S/'analysis_sources.json')}
    for p,h in sources.items():assert sha(R/p)==h
    manifest=load(S/'cycle_manifest.json');ready=load(S/'readiness.json');result=load(S/'result.json');count=0;max_input=max_time=0.
    with np.load(S/'identities.npz',allow_pickle=False) as z:allids=z['patient_ids'].copy();allvid=z['video_ids'].copy();eligible=z['eligible_rows'].copy()
    assert len(manifest)==203 and len(set(allids))==175;derived_eligible=[]
    for j,row in enumerate(manifest):
        with np.load(OLD/f'records/row_{j:04d}.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['front_raw'][-5:],z['last_raw'][:5]);raw=np.concatenate([z['front_raw'],z['last_raw'][5:]])
        scale=statistics.median([math.dist(raw[t,a,:2],raw[t,b,:2]) for t in range(243) for a,b in [(1,2),(2,3),(4,5),(5,6)]]);signal=[(float(raw[t,3,0])-float(raw[t,0,0]))/scale for t in range(243)]
        P,peaks,cycles=independent_detection(signal);assert P==row['right_detection']['period'] and peaks==row['right_detection']['peaks'] and cycles==row['right_detection']['cycles']
        choices={}
        for pos,start in [('front',0),('last',119)]:
            cc=[v for v in cycles if start<=v[0]<v[1]<=start+123];assert (len(cc) if P is not None else None)==row[pos+'_full_cycles'];pairs=[[a[0],a[1],b[1]] for a,b in zip(cc,cc[1:]) if a[1]==b[0]];best=min(pairs,key=lambda a:(abs((a[0]+a[2])/2-(start+61.5)),a[0])) if pairs else None;assert best==row['aligned_bounds'][pos];choices[pos]=best
        assert row['eligible']==all(x is not None for x in choices.values())
        if row['eligible']:
            derived_eligible.append(j);p=S/f'inputs/row_{j:04d}.npz';assert sha(p)==row['input_sha256']
            with np.load(p,allow_pickle=False) as z:
                for pos in ['front','last']:
                    a,b,c=row['aligned_bounds'][pos];ca,cb,cc=row['control_bounds'][pos];assert b-a==cb-ca and c-b==cc-cb
                    difference=z['aligned_'+pos+'_times']-z['control_'+pos+'_times'];err=float(np.max(abs(difference-(a-ca))));assert err<1e-10;max_time=max(max_time,err)
                    for arm in ['control','aligned']:
                        times=z[arm+'_'+pos+'_times'];bounds=row[arm+'_bounds'][pos];assert len(times)==124 and (np.diff(times)>0).all();close(times[0],bounds[0]);close(times[62],bounds[1]);close(times[-1],bounds[2]);expected=independent_input(raw,times);err=float(np.max(abs(expected-z[arm+'_'+pos])));assert err<2e-6;max_input=max(max_input,err)
            count+=1
    np.testing.assert_array_equal(eligible,derived_eligible);assert count==79 and len(set(allids[eligible]))==76
    with np.load(S/'features.npz',allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
    with np.load(BASE/'models.npz',allow_pickle=False) as z:m={k:z[k].copy() for k in z.files if k.startswith('fold')}
    rows=list(csv.DictReader((S/'predictions.csv').open(encoding='utf-8')));assert len(rows)==79*9;lookup={};labels={};folds={};ids={};vid=list(map(str,f['video_ids']));ix={v:i for i,v in enumerate(vid)};maxpred=0.
    for r in rows:
        a,p,v=r['condition'],r['position'],r['video_id'];key=(a,p,v);assert key not in lookup;lookup[key]=[float(r['prediction_L']),float(r['prediction_R'])];truth=[float(r['actual_L']),float(r['actual_R'])]
        if v in labels:assert labels[v]==truth and folds[v]==int(r['fold']) and ids[v]==r['patient_id']
        labels[v]=truth;folds[v]=int(r['fold']);ids[v]=r['patient_id'];assert ids[v]==str(f['patient_ids'][ix[v]])
        if p!='mean2':
            pre=f'fold{folds[v]}__';zz=(f[a+'_'+p][ix[v]]-m[pre+'scaler_mean'])/m[pre+'scaler_scale']
            for side in range(2):
                expected=math.fsum(float(x)*float(y) for x,y in zip(zz,m[pre+'coef'][side]))+float(m[pre+'intercept'][side]);err=abs(expected-lookup[key][side]);assert err<1e-8;maxpred=max(maxpred,err)
    people=sorted(set(ids.values()));assert len(people)==76
    def group(vals):return np.array([statistics.mean(vals[v] for v in vid if ids[v]==p) for p in people])
    metrics={}
    for a in ['original','control','aligned']:
        metrics[a]={}
        for p in ['front','last','mean2']:
            if p=='mean2':
                for v in vid:
                    for side in range(2):close(lookup[a,p,v][side],(lookup[a,'front',v][side]+lookup[a,'last',v][side])/2)
            error=group({v:statistics.mean(abs(q-y) for q,y in zip(lookup[a,p,v],labels[v])) for v in vid});close(error.mean(),result['metrics'][a]['person_mae'][p]);metrics[a]['mae_'+p]=error
            pp=group({v:statistics.mean(lookup[a,p,v]) for v in vid});close(statistics.stdev(pp),result['metrics'][a]['prediction_person_sd'][p])
        gap=group({v:statistics.mean(abs(x-y) for x,y in zip(lookup[a,'front',v],lookup[a,'last',v])) for v in vid});close(gap.mean(),result['metrics'][a]['score_gap']['mean']);metrics[a]['score_gap']=gap
        fg={}
        for v in vid:
            scale=m[f'fold{folds[v]}__scaler_scale'];d=(f[a+'_last'][ix[v]]-f[a+'_front'][ix[v]])/scale;fg[v]=math.sqrt(statistics.mean(float(x)**2 for x in d))
        fg=group(fg);close(fg.mean(),result['metrics'][a]['feature_gap']['mean']);metrics[a]['feature_gap']=fg
    for key,obj in result['comparisons'].items():
        candidate,reference=key.split('_vs_')
        for k,reported in obj.items():
            d=metrics[reference][k]-metrics[candidate][k];close(d.mean(),reported['reduction']);assert int((d>0).sum())==reported['people_improved'] and int((d<0).sum())==reported['people_worsened'];rng=np.random.default_rng(20261026);ci=np.percentile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[2.5,97.5]);np.testing.assert_allclose(ci,reported['bootstrap95'],atol=1e-8,rtol=0)
    output={'verified':True,'independent_right_detector_records':203,'independent_eligibility_records':79,'people':76,'independent_resampled_input_max':max_input,'matched_time_grid_translation_max':max_time,'prediction_rows':len(rows),'scalar_model_replay_max':maxpred,'person_metrics_verified':True,'bootstrap_recomputed':True,'source_hashes_checked':len(sources),'clinical_cycle_validation':False,'new_training':0}
    with (S/'verification.json').open('x',encoding='utf-8') as ff:json.dump(output,ff,indent=2)
    print(json.dumps(output))
if __name__=='__main__':main()
