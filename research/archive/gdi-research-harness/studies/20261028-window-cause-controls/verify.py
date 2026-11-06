import csv,json,hashlib,math,statistics
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;BASE=H/'studies/20261028-missing-augmentation-learning'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-8):assert abs(float(a)-float(b))<tol,(a,b)
def norm(x):
    observed=x[:,:,2]!=0;coords=x[:,:,:2];valid=coords[observed];lo=valid.min(0);hi=valid.max(0);out=x.astype(float).copy();out[:,:,:2]=(coords-(hi+lo)/2)*(2/max(hi-lo));return np.clip(out,-1,1).astype(np.float32)
def paths(x):
    xy=x[:,:,:2].astype(float);lengths=[]
    for t in range(124):lengths.append(statistics.median([math.dist(xy[t,a],xy[t,b]) for a,b in [(1,2),(2,3),(4,5),(5,6)]]))
    sizes=np.array([statistics.median(lengths[max(0,t-15):min(124,t+16)]) for t in range(124)])
    return xy[:,0],sizes

def angles(x):
    result=[]
    for a,b,c in [(1,2,3),(4,5,6)]:
        u=x[:,a,:2].astype(float)-x[:,b,:2];v=x[:,c,:2].astype(float)-x[:,b,:2];result.append(np.degrees(np.arctan2(np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0]),np.sum(u*v,axis=1))))
    return np.stack(result,1)
def main():
    result=load(S/'result.json');source=load(S/'sources.json')['sha256'];source.update(load(S/'analysis_sources.json'))
    for p,h in source.items():assert sha(R/p)==h
    with np.load(S/'features.npz',allow_pickle=False) as z:features={k:z[k].copy() for k in z.files}
    with np.load(BASE/'models.npz',allow_pickle=False) as z:models={k:z[k].copy() for k in z.files if k.startswith('fold')}
    meta=load(S/'diagnostic_rows.json');maxinput=maxangle=0.
    for i,item in enumerate(meta):
        path=S/f'records/row_{i:04d}.npz';assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as z:r={k:z[k].copy() for k in z.files}
        for p in ['front','last']:np.testing.assert_allclose(norm(r[p+'_raw']),r[p+'_input'],atol=2e-6,rtol=0)
        for kind in ['translation','scale','both']:
            for direction,a,b in [('FL','front','last'),('LF','last','front')]:
                raw=r[a+'_raw'];donor=r[b+'_raw'];rs,ss=paths(raw);rd,sd=paths(donor);hybrid=raw.astype(float).copy()
                for t in range(124):
                    for j in range(17):
                        factor=sd[t]/ss[t] if kind in ['scale','both'] else 1.;root=rd[t] if kind in ['translation','both'] else rs[t];hybrid[t,j,:2]=(raw[t,j,:2]-rs[t])*factor+root
                expect=norm(hybrid);missing=raw[:,:,2]==0;expect[missing,:2]=r[a+'_input'][missing,:2];actual=r[kind+'_'+direction+'_input'];delta=float(np.max(abs(expect-actual)));maxinput=max(maxinput,delta);assert delta<2e-6
                np.testing.assert_array_equal(actual[:,:,2],r[a+'_input'][:,:,2]);np.testing.assert_array_equal(actual[missing],r[a+'_input'][missing])
                ae=float(np.max(np.abs(angles(actual)-angles(r[a+'_input']))));maxangle=max(maxangle,ae);assert ae<1e-3
                np.testing.assert_array_equal(r[kind+'_'+direction],features[kind+'_'+direction][i])
        for p in ['front','last']:
            common=(r['front_input'][:,:,2]>0)&(r['last_input'][:,:,2]>0);actual=r['confidence_'+p+'_input'];np.testing.assert_array_equal(actual[:,:,:2],r[p+'_input'][:,:,:2]);np.testing.assert_array_equal(actual[:,:,2]==0,r[p+'_input'][:,:,2]==0)
            expected=np.sqrt(r['front_input'][:,:,2].astype(float)*r['last_input'][:,:,2]);np.testing.assert_allclose(actual[:,:,2][common],expected[common],atol=1e-7,rtol=0)
    rows=list(csv.DictReader((S/'predictions.csv').open(encoding='utf-8')));assert len(rows)==2030
    lookup={};ids={};folds={};videos=list(map(str,features['video_ids']));vi={v:i for i,v in enumerate(videos)};maxpred=0.
    for row in rows:
        v,k=row['video_id'],row['condition'];assert (v,k) not in lookup;lookup[v,k]=[float(row['prediction_L']),float(row['prediction_R'])];ids[v]=row['patient_id'];folds[v]=int(row['fold']);ix=vi[v];pre=f'fold{folds[v]}__'
        z=(features[k][ix]-models[pre+'scaler_mean'])/models[pre+'scaler_scale']
        for side in range(2):
            p=math.fsum(float(a)*float(b) for a,b in zip(z,models[pre+'coef'][side]))+float(models[pre+'intercept'][side]);err=abs(p-lookup[v,k][side]);assert err<1e-8;maxpred=max(maxpred,err)
    people=sorted(set(ids.values()));assert len(people)==175
    for fold in range(5):assert len({ids[v] for v in videos if folds[v]==fold})==35
    def pmean(by_video):return [statistics.mean(by_video[v] for v in videos if ids[v]==p) for p in people]
    original=pmean({v:statistics.mean(abs(a-b) for a,b in zip(lookup[v,'last'],lookup[v,'front'])) for v in videos});close(statistics.mean(original),result['original_score_gap']['mean'])
    maxdecomp=0.
    for kind in ['translation','scale','both']:
        nvals={};gvals={};cf={};cl={}
        for v in videos:
            ff,fl,lf,ll=(np.array(lookup[v,k]) for k in ['front',kind+'_FL',kind+'_LF','last']);n=((fl-ff)+(ll-lf))/2;g=((lf-ff)+(ll-fl))/2;maxdecomp=max(maxdecomp,float(np.max(abs(ll-ff-n-g))));nvals[v]=statistics.mean(abs(n));gvals[v]=statistics.mean(abs(g));cf[v]=statistics.mean(abs(lf-ff));cl[v]=statistics.mean(abs(ll-fl))
        target=result['crossovers'][kind]['score'];close(statistics.mean(pmean(nvals)),target['global_trajectory_component']['mean']);close(statistics.mean(pmean(gvals)),target['remaining_component']['mean']);close(statistics.mean(pmean(cf)),target['common_front_trajectory_gap']['mean']);close(statistics.mean(pmean(cl)),target['common_last_trajectory_gap']['mean'])
        diff=np.array(original)-(np.array(pmean(cf))+np.array(pmean(cl)))/2;close(float(diff.mean()),target['common_trajectory_vs_original']['reduction'])
    with np.load(S/'person_diagnostics.npz',allow_pickle=False) as z:
        for key in ['original_feature_gap','confidence_feature_gap','original_score_gap','confidence_score_gap']:
            expected={'original_feature_gap':result['original_feature_gap']['mean'],'original_score_gap':result['original_score_gap']['mean'],'confidence_feature_gap':result['confidence_common']['feature_gap']['mean'],'confidence_score_gap':result['confidence_common']['score_gap']['mean']}[key];close(float(z[key].mean()),expected)
    out={'verified':True,'records':203,'people':175,'prediction_rows':2030,'independent_transplant_input_max':maxinput,'normalized_input_knee_angle_max_deg':maxangle,'independent_scalar_model_prediction_max':maxpred,'score_decomposition_max':maxdecomp,'sources_unchanged':True,'source_hashes':len(source),'new_training':0,'limits':['bootstrap and all exploratory correlations not independently reimplemented','no video or 3D pose ground truth','synthetic hybrids not verified real camera trajectories']}
    with (S/'verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print(json.dumps(out))
if __name__=='__main__':main()
