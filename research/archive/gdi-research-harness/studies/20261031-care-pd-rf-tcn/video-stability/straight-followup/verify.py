"""Independent feature/metric reconstruction and frozen-checkpoint replay."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;VIDEO=HERE.parent;BASE=VIDEO.parent;FEATURE=BASE/'feature-analysis'
sys.path.insert(0,str(FEATURE));import study
import numpy as np
import joblib
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(x):(HERE/'verification.json').write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def independent_features(a):
    columns=[6*(j-1)+k for j in [1,2,4,5,7,8,16,17,18,19] for k in range(6)];x=a[:,columns];z=x-np.mean(x,axis=0);ac=[]
    for lag in range(5,41,5):
        v=z[:-lag];w=z[lag:];ac.append(float(np.sum(v*w)/max(np.sqrt(np.sum(v*v)*np.sum(w*w)),1e-12)))
    return np.concatenate([np.std(x,axis=0),np.mean(np.abs(x[1:]-x[:-1]),axis=0)*25,np.array(ac)]).astype(np.float32)
def person_metrics(y,p,person):
    e=p-y;v=[]
    for sid in np.unique(person):
        q=e[person==sid];v.append([np.abs(q).mean(),q.mean(),(np.abs(q)>=2).mean()])
    return np.mean(v,axis=0)
def main():
    with np.load(BASE/'data.npz') as data:d={k:data[k] for k in data.files}
    with np.load(HERE/'care-predictions.npz') as data:care={k:data[k] for k in data.files}
    with np.load(HERE/'care-features.npz') as data:cfeat={k:data[k] for k in data.files}
    report=json.loads((HERE/'care-analysis.json').read_text());errors=[];feature_errors=[];geometry_errors=[]
    assert report['data_sha256']==sha(BASE/'data.npz') and report['script_sha256']==sha(HERE/'analyze.py')
    assert care['seeds'].tolist()==study.SEEDS and len(np.unique(d['person']))==110
    for group,mask in [('eligible2',care['length']>=50),('rgb2',(care['length']>=50)&(care['cohort']!='BMCLab')),('common5',care['length']>=125)]:
        for name,entry in report['groups'][group].items():
            for subgroup,m in [('all',mask),('high',mask&(care['y']>=2)),('score3',mask&(care['y']==3))]:
                result=[]
                for j in range(5):
                    row=person_metrics(care['y'][m],care[name][j,m],care['person'][m]);result.append(row)
                    expected=[entry[subgroup]['seeds'][j][k] for k in ['MAE','bias','large_error']];errors.append(float(np.max(abs(row-expected))))
                    person=care['person'][m];v=care[name][j,m];fraction=np.mean([np.mean((v[person==sid]>=.5)&(v[person==sid]<1.5)) for sid in np.unique(person)]);errors.append(abs(float(fraction)-entry[subgroup]['seeds'][j]['score1_fraction']))
                expected=[entry[subgroup][k] for k in ['MAE','bias','large_error']];errors.append(float(np.max(abs(np.mean(result,0)-expected))))
                assert len(np.unique(care['person'][m]))==entry[subgroup]['people'] and m.sum()==entry[subgroup]['walks']
    for i in np.flatnonzero(care['length']>=50):
        x=d['x'][d['offsets'][i]:d['offsets'][i+1]]
        for name,a in [('first2',x[:50]),('center2',x[(len(x)-50)//2:(len(x)-50)//2+50])]:feature_errors.append(float(np.max(abs(independent_features(a)-cfeat[name][i]))))
    clips=json.loads((HERE/'clips.json').read_text());prot=json.loads((HERE/'protocol.json').read_text());assert len(clips)==56 and len(prot['subjects'])==14
    prior=Path('D:/gait-research-data/MoVi/experiments/walking-expansion');audit=json.loads((HERE/'availability.json').read_text())
    for path,digest in audit['source_sha256'].items():assert sha(Path(path))==digest
    for c in clips:
        path=prior/('subject-%02d'%c['subject'])/'heavy'/(c['camera']+'_800_comparison.npz');z=np.load(path);start,end=c['frames_one_based'];ids=np.flatnonzero((z['frames']>=start)&(z['frames']<=end));assert len(ids)==60 and end-start==59
        xy=z['reference_xy'][ids];assert ((xy[:,:,0]>=0)&(xy[:,:,0]<800)&(xy[:,:,1]>=0)&(xy[:,:,1]<600)&~z['reference_gap'][ids]).all()
        r=z['reference_3d'][ids];spans=[];full=[]
        for left,right in [(6,7),(0,1)]:
            a=r[:,right,:2]-r[:,left,:2];yaw=np.rad2deg(np.unwrap(np.arctan2(a[:,1],a[:,0])));spans.append(np.quantile(yaw,.95)-np.quantile(yaw,.05));full.append(np.ptp(yaw))
        pos=(r[:,6,:2]+r[:,7,:2])/2;disp=np.linalg.norm(pos[-1]-pos[0]);eff=disp/np.linalg.norm(np.diff(pos,axis=0),axis=1).sum();speed=disp/1000/(59/30)
        measured=[max(spans),max(full),eff,speed];expected=[c['geometry'][k] for k in ['heading_p90_span','heading_full_span','efficiency','speed_mps']];geometry_errors.append(float(np.max(abs(np.array(measured)-expected))))
        if c['condition']=='straight':assert max(spans)<=20 and max(full)<=30 and eff>=.98 and speed>=.2
        else:assert max(spans)>60
        partner=next(q for q in clips if q['subject']==c['subject'] and q['condition']==c['condition'] and q['camera']!=c['camera']);assert partner['frames_one_based']==c['frames_one_based']
    assert max(errors)<=1e-12 and max(feature_errors)<=1e-6 and max(geometry_errors)<=1e-10
    result=dict(passed=True,care_metric_max_error=max(errors),care_features_max_error=max(feature_errors),geometry_max_error=max(geometry_errors),paired_people=14,paired_clips=56,care_source_people=110,independent_person_weighting=True,geometry_sources_checked=60)
    if not (HERE/'movi-analysis.json').exists():
        result['movi_pending']=True;write(result);print(json.dumps(result));return
    collection=json.loads((HERE/'collection-verification.json').read_text());assert collection['passed'];movi=np.load(HERE/'movi-predictions.npz');mreport=json.loads((HERE/'movi-analysis.json').read_text());assert mreport['script_sha256']==sha(HERE/'analyze.py');mf=[];mr=[];mm=[]
    inference=json.loads((HERE/'received/main/complete.json').read_text());oldinference=json.loads((VIDEO/'received/main/complete.json').read_text());shared=0
    for name,digest in inference['source_sha256'].items():
        prefix='../care-pd-video-stability-20261031/'
        if name.startswith(prefix):assert digest==oldinference['source_sha256'][name[len(prefix):]];shared+=1
    assert shared>20
    for name,digest in collection['files'].items():assert sha(HERE/'received'/name)==digest
    for i,c in enumerate(clips):
        r=np.load(HERE/'received/main'/(c['key']+'_clean.npz'))['rotations'];six=r[...,:2].reshape(60,138).astype('float64');bins=np.floor(np.arange(60)/30*25+1e-8).astype(int);a=np.array([six[bins==b].mean(0) for b in range(50)],dtype='float32')
        mf.append(float(np.max(abs(independent_features(a)-movi['features'][i]))))
    for f in range(6):
        te=np.flatnonzero(d['fold']==f);tr=np.flatnonzero(d['fold']!=f);assert not set(d['person'][te])&set(d['person'][tr])
        for j,seed in enumerate(study.SEEDS):
            path=FEATURE/'received/fits/motion3d'/('fold%d_seed%d.joblib'%(f,seed));assert sha(path)==report['models'][str(path)]==mreport['models'][str(path)];model=joblib.load(path);model.n_jobs=1
            mr.append(float(np.max(abs(model.predict(movi['features'])-movi['predictions'][f*5+j]))))
            ids=te[care['length'][te]>=50];mr.append(float(np.max(abs(model.predict(cfeat['first2'][ids])-care['first2'][j,ids]))))
            path=BASE/'received/runs/main'/('fold%d'%f)/('seed%d'%seed)/'rf.joblib';assert sha(path)==mreport['models'][str(path)];model=joblib.load(path);model.n_jobs=1
            mr.append(float(np.max(abs(model.predict(movi['features698'])-movi['predictions698'][f*5+j]))))
    for label in ['motion128','original698']:
        for condition in ['straight','turn']:
            a=movi[label+'_'+condition];delta=a[:,1]-a[:,0];s=mreport['summary'][label][condition]
            expected=[s['view_abs'],s['view_bias'],s['score1_fraction']];actual=[abs(delta).mean(),delta.mean(),((a>=.5)&(a<1.5)).mean()];mm.append(float(np.max(abs(np.array(actual)-expected))))
            for j in range(5):mm.append(abs(float(abs(delta[:,j::5]).mean())-s['by_seed'][j]['view_abs']))
            included=mreport['summary']['detector_in_frame_subset']['people'];si=[prot['subjects'].index(v) for v in included];mm.append(abs(float(abs(delta[si]).mean())-mreport['summary']['detector_in_frame_subset'][label][condition]['view_abs']))
        a=movi[label+'_straight'];b=movi[label+'_turn'];v=(abs(a[:,1]-a[:,0]).mean(1)-abs(b[:,1]-b[:,0]).mean(1)).mean();mm.append(abs(float(v)-mreport['summary'][label]['paired_delta']['view_abs_difference']))
    assert max(mf)<=1e-6 and max(mr)<=1e-12 and max(mm)<=1e-12
    result.update(movi_pending=False,movi_features_max_error=max(mf),frozen_rf_replay_max_error=max(mr),movi_metrics_max_error=max(mm),checked_models=60,primary_models=30,initializations=5,shared_hmr2_asset_hashes_matched=shared,collection_sha256=sha(HERE/'collection-verification.json'),analysis_sha256=sha(HERE/'movi-analysis.json'),care_analysis_sha256=sha(HERE/'care-analysis.json'))
    write(result);print(json.dumps(result))
if __name__=='__main__':main()
