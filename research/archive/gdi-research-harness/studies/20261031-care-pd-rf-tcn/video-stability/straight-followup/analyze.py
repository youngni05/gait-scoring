"""Frozen-model duration and paired-view comparisons, no training."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;VIDEO=HERE.parent;BASE=VIDEO.parent;FEATURE=BASE/'feature-analysis'
sys.path.insert(0,str(FEATURE));import study
sys.path.insert(0,str(VIDEO));import analyze_video as adapter
import numpy as np
import joblib
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
COLS=np.array([6*(j-1)+k for j in study.MAJOR for k in range(6)])
def feat(x):return study.stats(x[:,COLS])[np.r_[60:120,240:308]]
def summarize_metrics(d,p,mask):
    result={}
    for name,select in [('all',mask),('high',mask&(d['y']>=2)),('score3',mask&(d['y']==3))]:
        rows=[study.metrics(d['y'][select],v[select],d['person'][select]) for v in p]
        for row,v in zip(rows,p):row['score1_fraction']=float(np.average((v[select]>=.5)&(v[select]<1.5),weights=study.weights(d['person'][select])))
        result[name]=dict(people=rows[0]['people'],walks=rows[0]['walks'],**{k:float(np.mean([r[k] for r in rows])) for k in ['MAE','bias','large_error','score1_fraction']},seeds=rows)
    return result
def care():
    d=study.load(True);original=np.load(FEATURE/'inputs.npz')['motion3d'];n=len(d['y']);length=np.diff(d['offsets']);idx2=np.flatnonzero(length>=50);idx5=np.flatnonzero(length>=125)
    xx={k:np.full((n,128),np.nan,dtype='float32') for k in ['first2','center2','first5']}
    for i in idx2:
        x=d['x'][d['offsets'][i]:d['offsets'][i+1]];xx['first2'][i]=feat(x[:50]);k=(len(x)-50)//2;xx['center2'][i]=feat(x[k:k+50])
        if len(x)>=125:xx['first5'][i]=feat(x[:125])
    np.savez_compressed(HERE/'care-features.npz',**xx,length=length)
    predictions={k:np.full((5,n),np.nan) for k in ['full','first2','center2','first5']};hashes={};replays=[]
    for f in range(6):
        te=np.flatnonzero(d['fold']==f);tr=np.flatnonzero(d['fold']!=f);assert not set(d['person'][te])&set(d['person'][tr])
        for j,seed in enumerate(study.SEEDS):
            path=FEATURE/'received/fits/motion3d'/('fold%d_seed%d.joblib'%(f,seed));m=joblib.load(path);m.n_jobs=1;hashes[str(path)]=sha(path)
            full=m.predict(original[te]);saved=np.load(FEATURE/'fits/motion3d'/('fold%d_seed%d.npz'%(f,seed)))
            assert np.array_equal(saved['indices'],te) and np.array_equal(saved['train'],tr)
            err=float(abs(full-saved['prediction']).max());assert err<=1e-12;replays.append(err);predictions['full'][j,te]=full
            for name in xx:
                ids=te[np.isfinite(xx[name][te]).all(1)];predictions[name][j,ids]=m.predict(xx[name][ids])
    groups={}
    for label,mask in [('eligible2',length>=50),('rgb2',(length>=50)&(d['cohort']!='BMCLab')),('common5',length>=125)]:
        groups[label]={k:summarize_metrics(d,p,mask) for k,p in predictions.items() if k!='first5' or label=='common5'}
    np.savez_compressed(HERE/'care-predictions.npz',**predictions,y=d['y'],person=d['person'],fold=d['fold'],cohort=d['cohort'],length=length,seeds=study.SEEDS)
    save('care-analysis.json',dict(passed=True,seeds=study.SEEDS,groups=groups,max_replay_error=max(replays),models=hashes,data_sha256=sha(BASE/'data.npz'),script_sha256=sha(Path(__file__))))
    print(json.dumps({g:{k:r['all']['MAE'] for k,r in rows.items()} for g,rows in groups.items()}),flush=True)
def band(a):return float(((a>=.5)&(a<1.5)).mean())
def viewmetrics(a):
    # a: [people, cameras, models]; cameras share exact synchronized frames.
    delta=a[:,1]-a[:,0]
    return dict(mean_score=float(a.mean()),score1_fraction=band(a),view_abs=float(abs(delta).mean()),view_bias=float(delta.mean()),
        positive_fraction=float((delta>0).mean()),mean_person_spread=float(a.mean(1).std(0).mean()),
        video_mean_min=float(a.mean(2).min()),video_mean_max=float(a.mean(2).max()),
        by_seed=[dict(seed=seed,mean_score=float(a[:,:,j::5].mean()),score1_fraction=band(a[:,:,j::5]),view_abs=float(abs(delta[:,j::5]).mean()),view_bias=float(delta[:,j::5].mean()),person_spread=float(a[:,:,j::5].mean(1).std(0).mean())) for j,seed in enumerate(study.SEEDS)])
def movi():
    prot=json.loads((HERE/'protocol.json').read_text());clips=json.loads((HERE/'clips.json').read_text());complete=json.loads((HERE/'received/main/complete.json').read_text())
    assert complete['passed'] and complete['frames']==3360 and len(complete['outputs'])==56
    for name in ['infer.py','protocol.json','clips.json']:assert sha(HERE/name)==complete['source_sha256'][name]
    rows={r['key']:r for r in complete['outputs']};xs=[];orig=[];rotations={}
    for clip in clips:
        key=clip['key']+'_clean';p=HERE/'received/main'/(key+'.npz');assert sha(p)==rows[key]['sha256'];z=np.load(p);r=z['rotations'];assert r.shape==(60,23,3,3) and int(z['fps'])==30 and not np.any(z['noise'])
        assert np.isfinite(r).all() and np.max(abs(r.transpose(0,1,3,2)@r-np.eye(3)))<=1e-5 and np.max(abs(np.linalg.det(r)-1))<=1e-5
        x=adapter.sequence_rotation(r);assert x.shape==(50,138);xs.append(feat(x));orig.append(study.stats(x));rotations[clip['key']]=r
    xs=np.stack(xs);orig=np.stack(orig);assert xs.shape==(56,128) and np.isfinite(xs).all()
    pred=[];pred698=[];hashes={};replay=[]
    for f in range(6):
        for seed in study.SEEDS:
            path=FEATURE/'received/fits/motion3d'/('fold%d_seed%d.joblib'%(f,seed));m=joblib.load(path);m.n_jobs=1;hashes[str(path)]=sha(path);a=m.predict(xs);assert np.max(abs(a-m.predict(xs)))<=1e-12;pred.append(a)
            path=BASE/'received/runs/main'/('fold%d'%f)/('seed%d'%seed)/'rf.joblib';m=joblib.load(path);m.n_jobs=1;hashes[str(path)]=sha(path);a=m.predict(orig);pred698.append(a)
    pred=np.array(pred);pred698=np.array(pred698);subjects=prot['subjects'];index={r['key']:i for i,r in enumerate(clips)};summary={};arrays={}
    for label,p in [('motion128',pred),('original698',pred698)]:
        summary[label]={}
        for cond in ['straight','turn']:
            a=np.array([[p[:,index['s%02d_%s_%s'%(sid,cam,cond)]] for cam in ['PG1','PG2']] for sid in subjects]);arrays[label+'_'+cond]=a;summary[label][cond]=viewmetrics(a)
        s=arrays[label+'_straight'];t=arrays[label+'_turn'];diff=abs(s[:,1]-s[:,0]).mean(1)-abs(t[:,1]-t[:,0]).mean(1)
        rng=np.random.RandomState(20261031);boot=diff[rng.randint(0,len(diff),(10000,len(diff)))].mean(1)
        summary[label]['paired_delta']=dict(view_abs_difference=float(diff.mean()),bootstrap_person_ci95=np.quantile(boot,[.025,.975]).tolist(),people_improved=int((diff<0).sum()),people=len(subjects),per_person=diff.tolist(),note='Descriptive person-bootstrap interval for this selected sample; not clinical external validation.')
    roi=json.loads((HERE/'roi-quality.json').read_text());ri=[subjects.index(s) for s in roi['people']]
    summary['detector_in_frame_subset']=dict(people=roi['people'],**{label:{cond:viewmetrics(arrays[label+'_'+cond][ri]) for cond in ['straight','turn']} for label in ['motion128','original698']})
    old=np.load(VIDEO/'scores.npz');oldkeys=old['keys'].tolist();oldsub=json.loads((VIDEO/'protocol.json').read_text())['selected'];common=[r['subject'] for r in oldsub if r['subject'] in subjects];summary['previous5_common']={}
    for label,pkey in [('motion128','predictions'),('original698','original_predictions')]:
        p=old[pkey];oldarr=np.array([[p[:,oldkeys.index('s%02d_%s_clean'%(sid,cam))] for cam in ['PG1','PG2']] for sid in common]);si=[subjects.index(sid) for sid in common]
        summary['previous5_common'][label]=dict(people=common,old5=viewmetrics(oldarr),straight2=viewmetrics(arrays[label+'_straight'][si]),turn2=viewmetrics(arrays[label+'_turn'][si]))
    perperson=[]
    for i,sid in enumerate(subjects):
        row=dict(subject=sid)
        for condition in ['straight','turn']:
            a=arrays['motion128_'+condition][i];row[condition]=dict(PG1=float(a[0].mean()),PG2=float(a[1].mean()),view_abs=float(abs(a[1]-a[0]).mean()),heading=next(r['geometry']['heading_p90_span'] for r in clips if r['subject']==sid and r['condition']==condition))
        perperson.append(row)
    np.savez_compressed(HERE/'movi-predictions.npz',features=xs,features698=orig,predictions=pred,predictions698=pred698,keys=[r['key'] for r in clips],subjects=subjects,**arrays)
    save('movi-analysis.json',dict(passed=True,summary=summary,per_person=perperson,models=hashes,script_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'protocol.json')))
    print(json.dumps(summary['motion128']),flush=True)
if __name__=='__main__':
    {'care':care,'movi':movi}[sys.argv[1]]()
