"""Frozen-model distribution, leaf-mixture and camera-path diagnostics."""
import sys,json,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;OUT=HERE/'diagnosis';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(BASE/'feature-analysis'));import study
import numpy as np
import joblib
from sklearn.metrics import pairwise_distances

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def average(x,p):return np.average(x,axis=0,weights=study.weights(p))
def scalar_stats(a):return dict(mean=float(np.mean(a)),min=float(np.min(a)),max=float(np.max(a)),sd=float(np.std(a)))
def simple(x):return np.stack([x[:,:60].mean(1),x[:,60:120].mean(1),x[:,120],x[:,124],x[:,127]],axis=1)
SUMMARY_NAMES=['rotation_component_std','rotation_component_change_per_second','ac_0.2s','ac_1s','ac_1.6s']
JOINT_NAMES=[study.JOINTS[j-1] for j in study.MAJOR]
NAMES=[f'{stat}:{joint}:R{r}{c}' for stat in ['std','change'] for joint in JOINT_NAMES for r in range(3) for c in range(2)]+['ac_'+str(lag) for lag in range(5,41,5)]
STAT_GROUP=np.array([0]*60+[1]*60+[2]*8)
JOINT_GROUP=np.array(list(np.repeat(np.arange(10),6))*2+[10]*8)

def main():
    start=time.time();verify=json.loads((HERE/'final-verification.json').read_text());assert verify['passed']
    assert sha(HERE/'analysis.json')==verify['analysis_sha256'] and sha(HERE/'scores.npz')==verify['scores_sha256']
    d=study.load(True);x=np.load(BASE/'feature-analysis/inputs.npz')['motion3d'];y=d['y'];person=d['person'];cohort=d['cohort'];fold=d['fold']
    vs=np.load(HERE/'scores.npz');ci=np.array([i for i,k in enumerate(vs['keys']) if k.endswith('_clean')]);v=vs['features'][ci];vkeys=vs['keys'][ci].tolist()
    assert len(ci)==10 and x.shape==(2953,128)
    cols=[6*(j-1)+k for j in study.MAJOR for k in range(6)]
    lengths=np.diff(d['offsets']);eligible=np.flatnonzero(lengths>=125);x5=np.full(x.shape,np.nan,dtype='float32')
    for i in eligible:
        a=d['x'][d['offsets'][i]:d['offsets'][i]+125][:,cols];x5[i]=study.stats(a)[np.r_[60:120,240:308]]
    del d['x']
    save('protocol.json',dict(scope='Post-hoc explanation of frozen predictions; no retraining, calibration, or camera correction fitted.',people_CARE=110,people_video=5,models=30,reference_5s='First125 existing25Hz frames of each eligible original walk; same-frame-length diagnostic only; original RF trained on full walks.',distribution='Person-equal summaries; training-fold-only weighted scale for nearest-neighbor distances. Compare raw features and five-second RGB-derived cohort subset.',leaf='Exact bootstrap multiplicity times original person weight; reconstruct label/cohort composition in each leaf and assert leaf prediction equals weighted label mean.',path='Exact root-to-leaf value differences allocated to split feature; tree-order-dependent explanation, not causal importance.',limitations='HMR2 vs WHAM is confounded with people, population, acquisition and motion. Five paired people cannot identify clinical accuracy or estimator-only causal effects.'))
    distributions=[]
    for window,a in [('full',x),('first5s',x5)]:
        valid=np.isfinite(a).all(1)
        for subset,smask in [('all',np.ones(len(y),bool)),('RGB',cohort!='BMCLab')]+[(str(c),cohort==c) for c in np.unique(cohort)]:
            for label in [-1,0,1,2,3]:
                m=valid&smask&((y==label) if label>=0 else True)
                if not m.any():continue
                distributions.append(dict(window=window,subset=subset,label=label,people=len(set(person[m])),walks=int(m.sum()),values=dict(zip(SUMMARY_NAMES,average(simple(a[m]),person[m]).tolist()))))
    for cam in ['PG1','PG2']:
        m=np.array([cam in k for k in vkeys]);distributions.append(dict(window='video5s',subset=cam,label=None,people=5,walks=5,values=dict(zip(SUMMARY_NAMES,simple(v[m]).mean(0).tolist()))))
    save('distributions.json',distributions)
    nearest=[];shifts=[];control=[];models=[];contribs=[];masses=[];cohort_masses=[];oof=np.full((5,len(y)),np.nan);oof5=np.full_like(oof,np.nan)
    cohort_names=np.unique(cohort);replay_errors=[];mass_errors=[];path_errors=[]
    for f in range(6):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        tw=study.weights(person[tr]);prior=np.bincount(y[tr].astype(int),weights=tw,minlength=4)/tw.sum()
        for window,a in [('full',x),('first5s',x5)]:
            for subset,mask in [('all',np.ones(len(y),bool)),('RGB',cohort!='BMCLab')]:
                train=tr[mask[tr]&np.isfinite(a[tr]).all(1)];test=te[mask[te]&np.isfinite(a[te]).all(1)]
                w=study.weights(person[train]);mu=np.average(a[train],axis=0,weights=w);sd=np.sqrt(np.average((a[train]-mu)**2,axis=0,weights=w));sd=np.maximum(sd,1e-6)
                zv=(v-mu)/sd;zt=(a[train]-mu)/sd;ze=(a[test]-mu)/sd
                dist=pairwise_distances(zv,zt)/np.sqrt(128);refdist=pairwise_distances(ze,zt).min(1)/np.sqrt(128)
                rw=study.weights(person[test]);train_people=np.unique(person[train]);personal=np.stack([dist[:,person[train]==p].min(1) for p in train_people],axis=1)
                for k,key in enumerate(vkeys):
                    closest=np.argsort(personal[k])[:10];records=[]
                    for c in closest:
                        choices=np.flatnonzero(person[train]==train_people[c]);idx=choices[np.argmin(dist[k,choices])];j=train[idx]
                        records.append(dict(person=str(person[j]),label=int(y[j]),cohort=str(cohort[j]),distance=float(dist[k,idx])))
                    nearest.append(dict(fold=f,window=window,subset=subset,key=key,nearest_distance=float(dist[k].min()),heldout_distance_percentile=float(np.average(refdist<=dist[k].min(),weights=rw)),ten_person_neighbor_label_mean=float(np.mean([r['label'] for r in records])),neighbors=records))
                shift=(v.mean(0)-mu)/sd;view=(v[1::2]-v[::2])/sd
                shifts.append(dict(fold=f,window=window,subset=subset,standardized_mean_shift=shift.tolist(),standardized_camera_difference=view.mean(0).tolist()))
        for si,seed in enumerate(study.SEEDS):
            path=BASE/'feature-analysis/received/fits/motion3d'/('fold%d_seed%d.joblib'%(f,seed));m=joblib.load(path);m.n_jobs=1
            assert sha(path)==json.loads((HERE/'analysis.json').read_text())['model_sha256'][str(path.relative_to(BASE))]
            pred=m.predict(v);err=float(abs(pred-vs['predictions'][f*5+si,ci]).max());assert err<=1e-12;replay_errors.append(err)
            oof[si,te]=m.predict(x[te]);te5=te[np.isfinite(x5[te]).all(1)];oof5[si,te5]=m.predict(x5[te5])
            leafmass=np.zeros((10,4));cmass=np.zeros((10,len(cohort_names)));contribution=np.zeros((10,128));root=0.;tree_preds=[];test_leafmass=np.zeros((len(te),4));test_tree_preds=[]
            for tree in m.estimators_:
                bt=np.random.RandomState(tree.random_state).randint(0,len(tr),len(tr));w=np.bincount(bt,minlength=len(tr))*tw
                leaves=tree.apply(x[tr]);size=tree.tree_.node_count;lookup=np.zeros((size,4));clookup=np.zeros((size,len(cohort_names)))
                for label in range(4):lookup[:,label]=np.bincount(leaves,weights=w*(y[tr]==label),minlength=size)
                for j,c in enumerate(cohort_names):clookup[:,j]=np.bincount(leaves,weights=w*(cohort[tr]==c),minlength=size)
                totals=lookup.sum(1);nz=totals>0;lookup[nz]/=totals[nz,None];clookup[nz]/=totals[nz,None]
                reached=tree.apply(v);lm=lookup[reached];tp=tree.predict(v);er=float(abs(lm@np.arange(4)-tp).max());assert er<=1e-10;mass_errors.append(er)
                leafmass+=lm/200;cmass+=clookup[reached]/200;tree_preds.append(tp)
                test_leafmass+=lookup[tree.apply(x[te])]/200;test_tree_preds.append(tree.predict(x[te]))
                value=tree.tree_.value[:,0,0];root+=value[0]/200
                for k,row in enumerate(v):
                    node=0
                    while tree.tree_.children_left[node]!=-1:
                        j=tree.tree_.feature[node];child=tree.tree_.children_left[node] if row[j]<=tree.tree_.threshold[node] else tree.tree_.children_right[node]
                        contribution[k,j]+=(value[child]-value[node])/200;node=child
                    assert node==reached[k]
            pe=float(abs(root+contribution.sum(1)-pred).max());assert pe<=1e-10;path_errors.append(pe)
            assert np.max(abs(leafmass.sum(1)-1))<1e-12
            assert np.max(abs(test_leafmass@np.arange(4)-oof[si,te]))<1e-10
            test_sd=np.std(test_tree_preds,axis=0)
            for label,mask in [('all',np.ones(len(te),bool)),('prediction_near1',(oof[si,te]>=.5)&(oof[si,te]<1.5))]+[(str(k),y[te]==k) for k in range(4)]:
                if mask.any():control.append(dict(fold=f,seed=seed,group=label,people=len(set(person[te][mask])),tree_sd_mean=float(average(test_sd[mask],person[te][mask])),label_mass=average(test_leafmass[mask],person[te][mask]).tolist()))
            for k,key in enumerate(vkeys):
                models.append(dict(fold=f,seed=seed,key=key,prediction=float(pred[k]),root_mean=float(root),prior_label_mass=prior.tolist(),leaf_label_mass=leafmass[k].tolist(),leaf_cohort_mass=dict(zip(cohort_names,cmass[k].tolist())),tree_prediction_sd=float(np.std(np.array(tree_preds)[:,k])),leaf_label_sd=float(np.sqrt(np.maximum(leafmass[k]@np.arange(4)**2-pred[k]**2,0))),selected_min_leaf=int(m.min_samples_leaf)))
            contribs.append(contribution);masses.append(leafmass);cohort_masses.append(cmass)
            print('inspected',f,seed,'seconds',round(time.time()-start,1),flush=True)
    assert np.isfinite(oof).all() and np.isfinite(oof5[:,eligible]).all()
    np.savez_compressed(OUT/'arrays.npz',contributions=np.stack(contribs),label_mass=np.stack(masses),cohort_mass=np.stack(cohort_masses),oof=oof,oof5=oof5,eligible=eligible,keys=np.array(vkeys),x5=x5)
    save('leaf-records.json',models);save('nearest.json',nearest);save('shifts.json',shifts);save('heldout-leaf-controls.json',control)
    # Group comparisons use one contribution for each model, camera and person.
    c=np.stack(contribs);camera_delta=c[:,1::2]-c[:,::2]
    attribution={}
    for title,ids,names in [('statistics',STAT_GROUP,['std','change','autocorrelation']),('joints',JOINT_GROUP,JOINT_NAMES+['autocorrelation'])]:
        attribution[title]=[dict(group=n,mean_camera_contribution=float(camera_delta[:,:,ids==j].sum(2).mean()),by_person=camera_delta[:,:,ids==j].sum(2).mean(0).tolist()) for j,n in enumerate(names)]
    labels=[]
    for window,pred,mask in [('full',oof,np.ones(len(y),bool)),('full_eligible5s',oof,np.isfinite(x5).all(1)),('first5s',oof5,np.isfinite(x5).all(1))]:
        for label in range(4):
            m=mask&(y==label);ps=pred[:,m];ww=study.weights(person[m]);labels.append(dict(window=window,label=label,people=len(set(person[m])),walks=int(m.sum()),prediction_mean=float(np.mean(np.average(ps,axis=1,weights=ww))),MAE=float(np.mean(np.average(abs(ps-y[m]),axis=1,weights=ww))),prediction_min=float(ps.min()),prediction_max=float(ps.max()),rounded_one_fraction=float(np.mean(np.average(np.floor(ps+.5)==1,axis=1,weights=ww)))))
    leaf_summary=[]
    for cam in ['PG1','PG2']:
        rows=[r for r in models if cam in r['key']];leaf_summary.append(dict(camera=cam,prediction_mean=float(np.mean([r['prediction'] for r in rows])),prior_mean=float(np.mean([np.array(r['prior_label_mass'])@np.arange(4) for r in rows])),root_mean=float(np.mean([r['root_mean'] for r in rows])),label_mass=np.mean([r['leaf_label_mass'] for r in rows],axis=0).tolist(),cohort_mass={n:float(np.mean([r['leaf_cohort_mass'][n] for r in rows])) for n in cohort_names},mean_tree_sd=float(np.mean([r['tree_prediction_sd'] for r in rows]))))
    top={}
    for window in ['full','first5s']:
        records=[r for r in shifts if r['window']==window and r['subset']=='RGB'];s=np.mean([r['standardized_mean_shift'] for r in records],axis=0)
        top[window]=[dict(feature=NAMES[j],standardized_difference=float(s[j])) for j in np.argsort(-abs(s))[:15]]
    control_summary=[]
    for g in ['all','prediction_near1','0','1','2','3']:
        rr=[r for r in control if r['group']==g];weights=[r['people'] for r in rr]
        control_summary.append(dict(group=g,tree_sd_mean=float(np.average([r['tree_sd_mean'] for r in rr],weights=weights)),label_mass=np.average([r['label_mass'] for r in rr],axis=0,weights=weights).tolist()))
    result=dict(passed=True,models=30,people_video=5,keys=vkeys,CARE_first5s_people=len(set(person[eligible])),CARE_first5s_walks=len(eligible),maximum_RF_replay_error=max(replay_errors),maximum_leaf_reconstruction_error=max(mass_errors),maximum_path_reconstruction_error=max(path_errors),leaf_summary=leaf_summary,heldout_tree_controls=control_summary,camera_attribution=attribution,CARE_prediction_by_label=labels,largest_RGB_distribution_shifts=top,nearest_summary=[dict(window=window,subset=subset,mean_heldout_distance_percentile=float(np.mean([r['heldout_distance_percentile'] for r in nearest if r['window']==window and r['subset']==subset])),mean_neighbor_label=float(np.mean([r['ten_person_neighbor_label_mean'] for r in nearest if r['window']==window and r['subset']==subset]))) for window in ['full','first5s'] for subset in ['all','RGB']],source_sha256={str(p.relative_to(BASE)):sha(p) for p in [HERE/'scores.npz',BASE/'data.npz',BASE/'feature-analysis/inputs.npz',Path(__file__)]},elapsed_seconds=time.time()-start)
    save('summary.json',result);print(json.dumps({k:result[k] for k in ['passed','leaf_summary','camera_attribution','nearest_summary']},indent=2))

if __name__=='__main__':main()
