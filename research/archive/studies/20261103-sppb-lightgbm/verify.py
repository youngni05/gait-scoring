"""Independent tree traversal, Ridge solve, label/split/hash/metric checks."""
from common import *
import csv,math,collections,itertools
import lightgbm as lgb
import openpyxl
from prepare import legacy_functions,IMU_NAMES

def tree_predict(node,x):
    if 'leaf_value' in node:return node['leaf_value']
    assert node['decision_type']=='<='
    value=x[node['split_feature']]
    left=node['default_left'] if np.isnan(value) else value<=node['threshold']
    return tree_predict(node['left_child'] if left else node['right_child'],x)

def labels(cohorts):
    sources={};subscore_unavailable=[];path=ROOT/'outputs/gait-public-access-20261015/OlderAdults44/01_demography_sppb_mmse.xlsx'
    w=openpyxl.load_workbook(path,read_only=True,data_only=True)
    yy={str(int(str(r[0]).split()[-1])):(r[7],r[2]) for r in list(w.active.values)[1:] if str(r[0]).startswith('Subject ')};w.close();sources[str(path)]=sha(path)
    for n,pp in cohorts.items():
        if n.startswith('older'):
            for p in pp:assert (p['sppb'],p['age'])==yy[p['id']]
    b=Path('D:/gait-research-data/D_P/larger-older-sppb-20261103')
    for ds,file,sheet,start,idcol,ycol,subcols in [('gait2care','Gait2Care_DDBB.xlsx','Week0',4,0,21,[17,18,20]),('gstride','GSTRIDE_DDBB.xlsx',None,3,1,28,[25,26,27])]:
        path=b/file;sources[str(path)]=sha(path);w=openpyxl.load_workbook(path,read_only=True,data_only=True);ws=w[sheet] if sheet else w.active
        table={str(r[idcol]):r for r in list(ws.values)[start:] if r[idcol]}
        for p in cohorts[ds]:
            r=table[p['id'].removeprefix('gstride-')];assert p['sppb']==float(r[ycol])
            try: subs=[float(r[j]) for j in subcols]
            except (ValueError,TypeError):
                # Preserve original eligibility: published total is retained when a subscore is unavailable.
                # Never convert 'Incapable' to zero or claim a sum check for these rows.
                subscore_unavailable.append(dict(dataset=ds,id=p['id'],raw_subscores=[r[j] for j in subcols]));continue
            assert p['sppb']==sum(subs)
        w.close()
    return dict(sources=sources,published_total_checked_all_people=True,subscore_sum_unavailable=subscore_unavailable)

def check_gdi(cohorts):
    prior=read(OLD/'frozen-verification.json');b=Path('D:/gait-research-data/D_P/older44-coordination-sppb-20261103')
    manifest_path=ROOT/'gdi-research-harness/studies/20261102-final223-frozen/models.json'
    assert sha(manifest_path)==prior['models_manifest_sha256'];manifest=read(manifest_path)
    assert sha(b/'features.npz')==prior['features_sha256'] and sha(b/'frozen-predictions.npz')==prior['predictions_sha256']
    outputs={}
    for r in prior['checks']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        path=b/'members'/(r['key']+'.npz');assert sha(path)==r['prediction_sha256']
        outputs[r['key']]=np.load(path)['prediction']
    fusions=[]
    for f,s in enumerate(manifest['selections']):
        bags={family:np.mean([outputs[f'{family}_{f}_{m["tag"]}'] for m in manifest['members'] if m['family']==family and m['fold']==f],axis=0) for family in ['CNN','STT']}
        base=s['cnn_weight']*bags['CNN']+(1-s['cnn_weight'])*bags['STT'];mean=base.mean(1)
        bilateral=outputs[f'bilateral_{f}'];delta=(1-s['fraction'])*(base[:,0]-base[:,1])+s['fraction']*(bilateral[:,0]-bilateral[:,1])
        fusions.append(np.stack([mean+delta/2,mean-delta/2],axis=1))
    result=np.mean(fusions,axis=0);np.testing.assert_allclose(result,np.load(b/'frozen-predictions.npz')['fusion'],rtol=0,atol=1e-12)
    z=np.load(b/'features.npz');trials=read(OLD/'trial-audit.json');maximum=0
    for p in cohorts['older37']:
        trial=[]
        for i,r in enumerate(trials):
            mask=(z['owner']==i)&(z['variant']=='primary')
            if r['id']==p['id'] and mask.any():trial.append(result[mask].mean())
        err=abs(float(np.mean(trial))-p['gdi']);maximum=max(maximum,err);assert err<1e-12
    return dict(checkpoint_hashes=len(prior['checks']),cached_member_outputs_reaggregated=len(outputs),people=37,max_aggregation_error=maximum,new_checkpoint_forward=False,prior_forward_replay_max=max(r['prior_replay_max'] for r in prior['checks']),geometry='previous audited virtual yaw-pan side camera; TOE/body25 equivalence remains unresolved; no RGB validity claimed')

def check_features(cohorts):
    cycles=read(BIG/'older-cycle-features.json');count=0
    for p in cohorts['older44']:
        cc=cycles[p['id']]
        for n,value in p['features'].items():
            if n=='speed':continue
            key=n.removesuffix('_mean').removesuffix('_asym_abs');means=[]
            for side in ['Left','Right']:
                a=[c['features'][key] for c in cc if c['side']==side and key in c['features']];means.append(math.fsum(a)/len(a))
            expected=sum(means)/2 if n.endswith('_mean') else abs(means[0]-means[1]);assert abs(expected-value)<1e-10
            count+=1
    # Different formula (atan2 of cross/dot) for knee angle in six real trials.
    parser=legacy_functions();raw=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-sppb-timeseries/raw');checks=0
    for p in cohorts['older44'][:6]:
        trial=cycles[p['id']][0]['file'];fr,q,ev,params,_=parser['load_trial'](raw/trial);local,_,_=parser['align'](q)
        shifted,_,_=parser['align'](q+np.array([12.,-7.,3.]))
        np.testing.assert_allclose(local[:,:,:2],shifted[:,:,:2],atol=1e-12,rtol=0,equal_nan=True)
        for c in [c for c in cycles[p['id']] if c['file']==trial and c['kinematic_valid']]:
            h,k,a=(0,2,4) if c['side']=='Left' else (1,3,5);mask=(fr/100>=c['t0'])&(fr/100<=c['t1']);v=q[mask]
            u=v[:,h]-v[:,k];w=v[:,a]-v[:,k];theta=180-np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u,w),axis=1),(u*w).sum(1)))
            expected=float(np.percentile(theta,95)-np.percentile(theta,5));assert abs(expected-c['features']['knee_ROM'])<1e-9;checks+=1
    # Re-read six source CSVs through the stdlib rather than pandas; recompute all expanded stride statistics.
    old={p['id']:p for p in read(LARGE/'people.json')};imu_checks=0
    for ds in ['gait2care','gstride']:
        for p in cohorts[ds][:3]:
            with open(old[p['id']]['source_file'],encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
            rr=[{k.replace(' ',''):float(v) if v.strip() else np.nan for k,v in row.items()} for row in rr]
            prefixes=['R_','L_'] if ds=='gait2care' else ['']
            valid={s:[row for row in rr if all(np.isfinite(row[s+k]) for k in ['CycleDuration[s]','StrideLength[m]','Per_Swing[%CD]','Stride_Velocity[m/s]','TurningAngle[degrees]']) and row[s+'CycleDuration[s]']>0 and row[s+'StrideLength[m]']>0 and 0<row[s+'Per_Swing[%CD]']<100 and row[s+'Stride_Velocity[m/s]']>0 and abs(row[s+'TurningAngle[degrees]'])<20] for s in prefixes}
            for j,key in enumerate(IMU_NAMES):
                arrays=[[r[s+key] for r in valid[s] if math.isfinite(r[s+key])] for s in prefixes];n=sum(map(len,arrays));means=[math.fsum(a)/len(a) for a in arrays]
                mean=math.fsum(v for a in arrays for v in a)/n;sd=math.sqrt(math.fsum((v-m)**2 for a,m in zip(arrays,means) for v in a)/(n-len(arrays)))
                assert abs(mean-p['features'][f'imu{j:02d}_mean'])<1e-8 and abs(sd-p['features'][f'imu{j:02d}_SD'])<1e-8;imu_checks+=2
                if len(arrays)==2:assert abs(abs(means[0]-means[1])-p['features'][f'imu{j:02d}_asym_abs'])<1e-8;imu_checks+=1
    return dict(person_aggregate_checks=count,independent_knee_cycle_checks=checks,independent_imu_statistics=imu_checks,measurement_accuracy='not validated against independent optical or clinical ground truth')

def main():
    cohorts=read(BIG/'cohorts.json');results=read(HERE/'comparison.json');fits=read(BIG/'fits.json');receipt=read(HERE/'fit-receipt.json');sources=read(HERE/'source-manifest.json')
    assert sha(HERE/'protocol.json')==receipt['protocol_sha256'] and sha(BIG/'cohorts.json')==receipt['cohorts_sha256'] and sha(HERE/'comparison.json')==receipt['comparison_sha256'] and sha(BIG/'fits.json')==receipt['fits_sha256']
    for path,h in sources.items():assert sha(path)==h,path
    label_sources=labels(cohorts);feature_checks=check_features(cohorts);gdi=check_gdi(cohorts)
    grouped=collections.defaultdict(list);max_error=0.;n_final=0
    for r in fits:
        n=r['cohort'];out=results[n];pp=cohorts[n];lookup={p['id']:i for i,p in enumerate(pp)};tr=np.array([lookup[i] for i in r['train']]);te=np.array([lookup[i] for i in r['evaluation']]);folds=np.array(out['folds']);outer=r['outer']
        assert len(set(r['train'])&set(r['evaluation']))==0
        if r['inner'] is not None:
            assert all(folds[tr]!=outer) and all(folds[te]!=outer)
            outer_tr=np.flatnonzero(folds!=outer);expected=list(KFold(3,shuffle=True,random_state=20261031).split(outer_tr))[r['inner']]
            np.testing.assert_array_equal(tr,outer_tr[expected[0]]);np.testing.assert_array_equal(te,outer_tr[expected[1]])
        else:
            np.testing.assert_array_equal(tr,np.flatnonzero(folds!=outer));np.testing.assert_array_equal(te,np.flatnonzero(folds==outer))
        y=np.array(out['y']);grouped[(n,r['condition'],outer,r['candidate'])].append(r)
        if 'model' not in r:continue
        fn=out['feature_names'][r['condition']];X=np.array([[p['gdi'] if f=='predicted_GDI' else p['features'][f] for f in fn] for p in pp],float)
        assert sha(r['model'])==r['model_sha256']
        if r['kind']=='ridge':
            a=read(r['model']);mu=np.mean(X[tr],0);scale=np.std(X[tr],0);scale[scale==0]=1
            Z=(X[tr]-mu)/scale;coef=np.linalg.solve(Z.T@Z+r['params']['alpha']*np.eye(X.shape[1]),Z.T@(y[tr]-y[tr].mean()))
            np.testing.assert_allclose(a['mean'],mu,rtol=0,atol=1e-12);np.testing.assert_allclose(a['scale'],scale,rtol=0,atol=1e-12);np.testing.assert_allclose(a['coef'],coef,rtol=0,atol=1e-10)
            p=(X[te]-mu)/scale@coef+y[tr].mean()
        else:
            model=lgb.Booster(model_file=r['model']);direct=model.predict(X[te],num_threads=2);dump=read(Path(r['model']).with_suffix('.trees.json'))
            p=np.array([sum(tree_predict(t['tree_structure'],row) for t in dump['tree_info']) for row in X[te]])
            np.testing.assert_allclose(p,direct,rtol=0,atol=1e-10)
        max_error=max(max_error,float(np.max(abs(p-r['prediction']))));np.testing.assert_allclose(p,r['prediction'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(p,np.array(out['predictions'][r['condition']])[te],rtol=0,atol=1e-10);n_final+=1
    for n,out in results.items():
        pp=cohorts[n];assert out['ids']==[p['id'] for p in pp] and out['y']==[p['sppb'] for p in pp]
        y=np.array(out['y'])
        for s in out['selection']:
            losses=[]
            for ci in range(len(s['inner_MAE'])):
                rr=[r for r in grouped[(n,s['condition'],s['outer'],ci)] if r['inner'] is not None];assert len(rr)==3
                err=[abs(pred-y[out['ids'].index(pid)]) for r in rr for pid,pred in zip(r['evaluation'],r['prediction'])]
                losses.append(math.fsum(err)/len(err))
            np.testing.assert_allclose(losses,s['inner_MAE'],rtol=0,atol=1e-12);assert s['candidate']==int(np.argmin(losses))
            final=[r for r in fits if r['cohort']==n and r['condition']==s['condition'] and r['outer']==s['outer'] and r['inner'] is None];assert len(final)==1 and final[0]['params']==s['params']
        for condition,pred in out['predictions'].items():
            p=np.array(pred);err=p-y;m=out['metrics'][condition]
            assert abs(math.fsum(abs(float(e)) for e in err)/len(err)-m['MAE'])<1e-12
            assert abs(math.sqrt(math.fsum(float(e)**2 for e in err)/len(err))-m['RMSE'])<1e-12
            assert abs(float(err.mean())-m['bias'])<1e-12
            assert abs(1-float(err@err)/float((y-y.mean())@(y-y.mean()))-m['R2'])<1e-12
            if np.ptp(p)>0:assert abs(float(spearmanr(y,p).statistic)-m['spearman'])<1e-12
            else:assert m['spearman'] is None
            assert m['within1']==sum(abs(float(e))<=1 for e in err)/len(err) and m['errors_gt2']==sum(abs(float(e))>2 for e in err)
            assert m['TP']==sum(a<=8 and b<=8 for a,b in zip(y,p)) and m['FP']==sum(a>8 and b<=8 for a,b in zip(y,p))
            assert m['FN']==sum(a<=8 and b>8 for a,b in zip(y,p)) and m['TN']==sum(a>8 and b>8 for a,b in zip(y,p))
            for band,lo,hi in [('0-4',0,4),('5-8',5,8),('9-10',9,10),('11-12',11,12)]:
                ix=(y>=lo)&(y<=hi);assert m['bands'][band]['n']==int(ix.sum())
                if ix.any():assert abs(abs(err[ix]).mean()-m['bands'][band]['MAE'])<1e-12 and abs(err[ix].mean()-m['bands'][band]['bias'])<1e-12
        indices=np.random.RandomState(20261103).randint(0,len(y),(10000,len(y)))
        for name,c in out['contrasts'].items():
            a,b=name.split('-');d=abs(np.array(out['predictions'][a])-y)-abs(np.array(out['predictions'][b])-y)
            q=np.percentile(np.take(d,indices).mean(axis=1),[2.5,97.5]);np.testing.assert_allclose(q,c['CI95'],atol=1e-12,rtol=0);assert abs(d.mean()-c['MAE_difference'])<1e-12
        print('verified',n,'n=',len(y),flush=True)
    assert n_final==85 and len(fits)==3340
    summary=dict(status='PASS_WITH_MEASUREMENT_AND_GENERALIZATION_LIMITS',source_hashes=len(sources),label_sources=label_sources,fit_split_records=len(fits),saved_final_models_replayed=n_final,independent_prediction_max_error=max_error,feature_checks=feature_checks,gdi=gdi,protocol_sha256=sha(HERE/'protocol.json'),comparison_sha256=sha(HERE/'comparison.json'),code_sha256=sha(Path(__file__)),notes=['Inner model selection recomputed from all saved inner predictions; inner trees not independently retrained.','All final trees reloaded and independently traversed; all final Ridge fits independently solved.','Frozen GDI cached outputs reaggregated and checkpoint hashes verified; no new CP-model forward pass in this run.','No independent sensor/marker/RGB measurement accuracy validation.'])
    save(HERE/'verification.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
