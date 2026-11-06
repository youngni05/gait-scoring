"""Frozen four-arm Huber point-loss comparison; full audit gate, no neural-network training or outer selection."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, os, traceback, time, uuid
import numpy as np

H=Path(__file__).resolve().parent; S=H.parent; ROOT=S.parents[1]
O=H/'ensemble/analysis'
CACHE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-huber-point692/ensemble/cache')
PLAN_SHA='4a1320fabe90278502cddad57f0dd71b0783a999672d01ebc8b49f06261deee3'

def resolve_path(p):
    value=str(p).replace('\\','/')
    prefix='C:/Users/jun/Desktop/D_P/'
    return Path('D:/Projects/D_P/'+value[len(prefix):]) if value.startswith(prefix) else Path(p)
def read(p): return json.loads(resolve_path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()
def save(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(12):
        try: t.replace(p); return
        except PermissionError:
            if attempt==11: raise
            time.sleep(.1*(attempt+1))
def arrays(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def mae(y,p,ids): return float(np.mean([abs(p[ids==i]-y[ids==i]).mean() for i in np.unique(ids)]))

def main():
    O.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    process=dict(pid=os.getpid(),active=True,started_at=datetime.datetime.now().astimezone().isoformat(),script_sha256=sha(__file__),protocol_sha256=sha(H/'protocol.json'),new_training=False)
    save(O/'process-receipt.json',process)
    try: execute(process)
    except BaseException:
        process['failed']=True; save(O/'execution-failure.json',dict(traceback=traceback.format_exc())); raise
    finally:
        process.update(active=False,ended_at=datetime.datetime.now().astimezone().isoformat()); save(O/'process-receipt.json',process)

def execute(process):
    assert sha(H/'protocol.json')==PLAN_SHA
    cfg=read(H/'protocol.json'); arms=cfg['evaluation']['ensemble_candidates']
    assert arms==['oldMSE','oldRank1','HuberOnly','HuberRank1']
    assert cfg['training_seed']==20261028 and cfg['point_loss']['search'] is False
    grid=[i/20 for i in range(21)]
    assert cfg['evaluation']['weight_grid']==grid
    assert cfg['evaluation']['bootstrap']==dict(unit='paired person',replicates=10000,seed=20261101)
    reg=read(ROOT/'control/gpu-watch/registry.json'); assert reg['thread_id']==cfg['owner_thread'] and reg['automation_id']==cfg['automation_id']
    sources={str(H/'protocol.json'):PLAN_SHA,str(H/'compare_completed.py'):sha(__file__)}
    def evidence(p,digest=None):
        p=resolve_path(p); got=sha(p)
        if digest is not None: assert got==digest,str(p)
        sources[str(p)]=got; return p
    # Keep the independently checked selection routine and re-evaluate its grid below.
    selector=evidence(S/'ranksim-ensemble692/select_weight.py')
    spec=importlib.util.spec_from_file_location('seed_inner_weight',selector); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    choose=module.choose_weight
    newrow=next(e for e in reg['experiments'] if e['id']==cfg['id'])
    oldrow=next(e for e in reg['experiments'] if e['id']=='cnn-ranksim-692')
    verification=read(evidence(H/'analysis/full-verification.json'))
    assert verification['qualified_full_verification_passed'] and verification['fits']==50 and verification['final_cpu_replays']==10 and verification['max_model_replay_error']<1e-4
    newreceipt=read(evidence(H/'checks/full-receipt.json',newrow['retrieval_receipt_sha256']))
    assert newreceipt['passed'] and newreceipt['source_succeeded']
    assert verification['receipt_sha256']==sha(H/'checks/full-receipt.json')
    oldstudy=S/'cnn-ranksim692'
    oldreport=read(evidence(oldstudy/'analysis/comparison.json',oldrow['reported_digest']))
    assert oldreport['qualified_full_verification_passed'] and oldreport['fits']==50 and oldreport['final_cpu_replays']==10
    oldreceipt=read(evidence(oldstudy/'checks/closeout-receipt.json',oldrow['retrieval_receipt_sha256']))
    assert oldreceipt['server_local_sha256_matched'] and newreceipt['server_local_sha256_matched']
    paths={'old':Path(oldreceipt['archive']).parent/'performance-review-originals','new':Path(newreceipt['archive']).parent/'performance-review-originals'}
    evidence(paths['old']/'manifest.json',cfg['source_manifest_sha256'])
    evidence(paths['new']/'manifest.json',newrow['manifest_sha256'])
    rawarms={'oldMSE':('old','mse'),'oldRank1':('old','ranksim'),'HuberOnly':('new','mse'),'HuberRank1':('new','ranksim')}
    base_report=read(evidence(S/'inner-bagging692/analysis/comparison.json'))
    base=arrays(evidence(base_report['predictions_path'],base_report['predictions_sha256']))
    keys=['y','person_id','video_id','fold'];d={k:base[k] for k in keys}; y=d['y']; ids=d['person_id']; people=np.unique(ids)
    assert len(people)==692 and len(y)==1061 and abs(mae(y,base['bagged'],ids)-6.515816569178735)<1e-10
    md=S/'mean-difference692/analysis';es=S/'cnn-stt-ensemble692/analysis'
    selection=read(evidence(md/'selection.json')); proof=read(evidence(md/'verification.json'))
    assert proof['passed'] and proof['selection_sha256']==sha(md/'selection.json') and proof['predictions_sha256']==sha(md/'predictions.npz')
    old=read(evidence(es/'selection.json')); oldproof=read(evidence(es/'verification.json'))
    assert oldproof['passed'] and oldproof['all_20_base_fit_person_boundaries']
    for p,h in selection['source_hashes'].items(): evidence(p,h)
    for p,h in old['source_hashes'].items(): evidence(S/p,h)
    def boundary(ident,f,j):
        tr=(d['fold']!=f)&(d['fold']!=j); va=d['fold']==j; te=d['fold']==f
        for field,mask in [('train_people',tr),('validation_people',va),('outer_evaluation_people',te)]: assert ident[field]==np.unique(ids[mask]).tolist(),(field,f,j)
        assert not set(ident['train_people'])&set(ident['validation_people'])
        assert not (set(ident['train_people'])|set(ident['validation_people']))&set(ident['outer_evaluation_people'])
        return tr,va,te
    try: boundary(dict(train_people=['injected'],validation_people=[],outer_evaluation_people=[]),0,1)
    except AssertionError: pass
    else: raise AssertionError('Injected leakage accepted')
    all_pred={a:np.full_like(y,np.nan) for a in arms}; standalone={a:np.full_like(y,np.nan) for a in rawarms}; raw_standalone={a:np.full_like(y,np.nan) for a in rawarms}
    selections=[]; correlations=[]
    for f in range(5):
        save(O/'progress.json',dict(phase='inner_prediction_provenance',outer=f,completed_outer=f))
        sf=selection['folds'][f]; assert sf['outer']==f
        v=arrays(evidence(md/('inner_outer%d.npz'%f),sf['inner_sha256'])); ix=v['row_index']; np.testing.assert_array_equal(ix,np.flatnonzero(d['fold']!=f))
        for k in keys: np.testing.assert_array_equal(v[k],d[k][ix])
        e=arrays(evidence(es/('inner_outer%d.npz'%f),old['folds'][f]['inner_file_sha256']));np.testing.assert_array_equal(e['row_index'],ix)
        np.testing.assert_allclose(v['base'],sf['cnn_weight']*e['cnn']+(1-sf['cnn_weight'])*e['stt'],rtol=0,atol=1e-12)
        diff=(v['bilateral'][:,0]-v['bilateral'][:,1])-(v['base'][:,0]-v['base'][:,1])
        inner_base=v['base']+np.column_stack([sf['fraction']*diff/2,-sf['fraction']*diff/2])
        inner={a:np.full_like(inner_base,np.nan) for a in rawarms}
        for j in range(5):
            if f==j: continue
            pos=v['fold']==j; va=d['fold']==j;np.testing.assert_array_equal(ix[pos],np.flatnonzero(va))
            for folder in [S/'cnn-multiwindow695/received/results'/('fold%d/inner%d'%(f,j)),S/'stt692/multiwindow/analysis/unpacked/results'/('fold%d/inner%d'%(f,j)),S/'stt692/bilateral/analysis/unpacked/results'/('fold%d/inner%d'%(f,j))]: boundary(read(evidence(folder/'identity.json')),f,j)
            for a,(seed,variant) in rawarms.items():
                folder=paths[seed]/'runs/main'/variant/('fold%d/inner%d'%(f,j)); ident=read(evidence(folder/'identity.json'));tr,valid,_=boundary(ident,f,j)
                assert ident['purpose']=='inner_epoch_selection' and ident['epochs']==100 and ident['outer_fold']==f and ident['inner_fold']==j
                done=read(evidence(folder/'fit_complete.json'));assert done['identity']==ident
                evidence(folder/'latest.pt',done['checkpoint_sha256']);curve=read(evidence(folder/'curve.json',done['curve_sha256']));row=curve[-1];assert row['epoch']==100 and len(curve)==100
                z=arrays(evidence(folder/'raw_predictions/epoch0100.npz',row['raw_predictions_sha256']));p=z['validation_raw']*row['calibration'][0]+row['calibration'][1]
                assert p.shape==y[valid].shape and abs(mae(y[valid],p,ids[valid])-row['validation_person_mae'])<1e-10
                assert hashlib.sha256(y[tr].tobytes()).hexdigest()==ident['train_y_sha'] and hashlib.sha256(ids[tr].tobytes()).hexdigest()==ident['train_ids_sha']
                inner[a][pos]=p
        assert not set(v['person_id'])&set(ids[d['fold']==f])
        outer_add={}
        for a,(seed,variant) in rawarms.items():
            folder=paths[seed]/'runs/main'/variant/('fold%d'%f);done=read(evidence(folder/'complete.json'));outer=arrays(evidence(folder/'predictions.npz',done['prediction_sha256']));refit=read(evidence(folder/'refit/identity.json'))
            assert refit['train_people']==np.unique(ids[d['fold']!=f]).tolist() and refit['outer_evaluation_people']==np.unique(ids[d['fold']==f]).tolist() and refit['validation_people']==[]
            oi=outer['row_index'];np.testing.assert_array_equal(oi,np.flatnonzero(d['fold']==f))
            for k in keys:np.testing.assert_array_equal(outer[k],d[k][oi])
            outer_add[a]=outer['predictions'][:,0];standalone[a][oi]=outer_add[a];raw_standalone[a][oi]=outer['predictions_raw'][:,0]
        for a in arms:
            assert np.isfinite(inner[a]).all()
            chosen=choose(v['y'],inner_base,inner[a],v['person_id'],grid);w=chosen['rank_weight']
            scores=[mae(v['y'],(1-q)*inner_base+q*inner[a],v['person_id']) for q in grid]
            assert w==next(q for q,score in zip(grid,scores) if score<=min(scores)+1e-12)
            assert choose(v['y'][::-1],inner_base[::-1],inner[a][::-1],v['person_id'][::-1],grid)['rank_weight']==w
            poisoned=y.copy();poisoned[d['fold']==f]=1e9;assert choose(poisoned[ix],inner_base,inner[a],v['person_id'],grid)['rank_weight']==w
            all_pred[a][oi]=(1-w)*base['bagged'][oi]+w*outer_add[a]
            selections.append(dict(outer=f,arm=a,additional_weight=w,inner_scores=scores,inner_people=len(np.unique(v['person_id'])),outer_labels_used=False))
            correlations.append(dict(outer=f,arm=a,signed_error_correlation=float(np.corrcoef((inner_base-v['y']).ravel(),(inner[a]-v['y']).ravel())[0,1]),prediction_disagreement=float(np.abs(inner_base-inner[a]).mean())))
        np.savez_compressed(CACHE/('inner_outer%d.npz'%f),**{k:v[k] for k in keys+['row_index']},base=inner_base,**inner);evidence(CACHE/('inner_outer%d.npz'%f));print('HUBER_ENSEMBLE_OUTER',f,flush=True)
    oldens=read(evidence(S/'ranksim-ensemble692/analysis/comparison.json'));original=arrays(evidence(oldens['predictions_path'],oldens['predictions_sha256']))
    for k in keys:np.testing.assert_array_equal(original[k],d[k])
    np.testing.assert_array_equal(all_pred['oldRank1'],original['ranksim']);assert abs(mae(y,all_pred['oldRank1'],ids)-6.476734533865628)<1e-10
    seedrow=next(e for e in reg['experiments'] if e['id']=='cnn-ranksim-seed-692')
    assert seedrow['review_state']=='reported'
    seed_report=read(evidence(S/'cnn-ranksim-seed692-v2/analysis/combined-comparison.json',seedrow['reported_digest']))
    seed_cmp=read(evidence(S/'cnn-ranksim-seed692-v2/ensemble/analysis/comparison.json',cfg['prior_comparison_sha256']))
    assert seed_cmp['passed'] and seed_cmp['qualified_full_verification_passed']
    seed_preds=arrays(evidence(seed_cmp['predictions_path'],seed_cmp['predictions_sha256']))
    for k in keys:np.testing.assert_array_equal(seed_preds[k],d[k])
    fixed_two_seed=seed_preds['add_mean_old_new_rank1']
    assert abs(mae(y,fixed_two_seed,ids)-6.476387909837368)<1e-10
    pm=np.array([y[ids==p].mean() for p in people]);high=pm>=90
    def perperson(v): return np.array([v[ids==i].mean() for i in people])
    def metrics(p):
        e=p-y;err=perperson(abs(e));bias=perperson(e)
        out=dict(mae=float(err.mean()),bias=float(bias.mean()),high90=dict(people=int(high.sum()),mae=float(err[high].mean()),bias=float(bias[high].mean())),fold_mae=[float(np.mean([abs(e[ids==i]).mean() for i in people if d['fold'][ids==i][0]==f])) for f in range(5)],bands={})
        for w in [10,20]:out['bands'][str(w)]=float(perperson(np.searchsorted(np.arange(0,161,w),p,side='right')==np.searchsorted(np.arange(0,161,w),y,side='right')).mean())
        out['subgroups']={}
        for label,mask in [('person_mean100',pm>=100),('person_mean60below',pm<60)]:out['subgroups'][label]=dict(people=int(mask.sum()),mae=float(err[mask].mean()),bias=float(bias[mask].mean()))
        for t in [90,100]:
            maes=[];biases=[];n=0;records=0
            for i in people:
                mask=y[ids==i]>=t;ee=e[ids==i][mask]
                if len(ee):maes.append(abs(ee).mean());biases.append(ee.mean());n+=len(ee);records+=int(mask.any(1).sum())
            out['subgroups']['observation_ge%d'%t]=dict(people=len(maes),records=records,side_observations=n,mae=float(np.mean(maes)),bias=float(np.mean(biases)))
        return out,err,bias
    predictions={'existing_best':base['bagged'],'existing_rank_ensemble':original['ranksim'],'existing_two_seed_rank_ensemble':fixed_two_seed,**{'add_'+a:p for a,p in all_pred.items()},**{'standalone_'+a:p for a,p in standalone.items()},**{'raw_'+a:p for a,p in raw_standalone.items()}}
    models={};errors={};biases={}
    for a,p in predictions.items():assert np.isfinite(p).all();models[a],errors[a],biases[a]=metrics(p)
    rng=np.random.default_rng(20261101);draw=rng.integers(0,len(people),(10000,len(people)));hd=rng.integers(0,int(high.sum()),(10000,int(high.sum())))
    def paired(a,b):
        delta=errors[a]-errors[b];dh=delta[high];bh=(biases[a]-biases[b])[high]
        return dict(delta=float(delta.mean()),bootstrap95=np.quantile(delta[draw].mean(1),[.025,.975]).tolist(),high90_delta=float(dh.mean()),high90_bootstrap95=np.quantile(dh[hd].mean(1),[.025,.975]).tolist(),high90_bias_delta=float(bh.mean()),high90_bias_delta_bootstrap95=np.quantile(bh[hd].mean(1),[.025,.975]).tolist(),improved_folds=int(np.sum(np.array(models[a]['fold_mae'])<np.array(models[b]['fold_mae']))))
    comparisons={}
    for a in arms:
        for b in ['existing_best','existing_rank_ensemble','existing_two_seed_rank_ensemble']:comparisons['add_'+a+'_vs_'+b]=paired('add_'+a,b)
    for a,b in [('standalone_HuberRank1','standalone_HuberOnly'),('standalone_oldRank1','standalone_oldMSE'),('standalone_HuberRank1','standalone_oldRank1'),('standalone_HuberOnly','standalone_oldMSE'),('add_HuberRank1','add_HuberOnly')]:comparisons[a+'_vs_'+b]=paired(a,b)
    stability={}
    for a,b in [('oldRank1','HuberRank1'),('oldMSE','HuberOnly')]:stability[a+'_vs_'+b]=dict(person_absolute_error_correlation=float(np.corrcoef(errors['standalone_'+a],errors['standalone_'+b])[0,1]),signed_error_correlation=float(np.corrcoef((standalone[a]-y).ravel(),(standalone[b]-y).ravel())[0,1]),prediction_disagreement=float(abs(standalone[a]-standalone[b]).mean()))
    primary=comparisons['add_HuberRank1_vs_existing_rank_ensemble'];m=models['add_HuberRank1'];b=models['existing_rank_ensemble']
    promotion=primary['bootstrap95'][1]<0 and primary['improved_folds']>=4 and m['high90']['mae']<=b['high90']['mae'] and abs(m['high90']['bias'])<=abs(b['high90']['bias'])
    dest=CACHE/'evaluation.npz';np.savez_compressed(dest,**d,**predictions);evidence(dest)
    save(O/'selection.json',dict(protocol_sha256=PLAN_SHA,folds=selections,inner_error_correlations=correlations,source_hashes=sources,no_cross_arm_winner_selected=True))
    report=dict(passed=True,qualified_full_verification_passed=True,people=692,records=1061,models=models,comparisons=comparisons,primary=primary,joint_improvement_gate_passed=bool(promotion),matched_loss_diagnostics=stability,predictions_path=str(dest),predictions_sha256=sha(dest),selection_sha256=sha(O/'selection.json'),protocol_sha256=PLAN_SHA,full_verification_sha256=sha(H/'analysis/full-verification.json'),bootstrap_replicates=10000,bootstrap_seed=20261101,outer_label_selection=False,global_OOF_meta_split=False,new_training=False,limitations=['Same repeatedly explored development cohort; no protected final or external clinical evaluation','One fixed matched seed; this loss comparison does not establish initialization stability','Inner single baseline versus outer bagged baseline','Original CNN/STT/bilateral fractions are meta-fit on outer-training people','Inner epoch100 versus individually selected outer refit epoch; fixed previous two-seed ensemble is a comparator only'])
    save(O/'comparison.json',report)
    save(O/'verification.json',dict(passed=True,full_model_audit_required=True,all_input_hashes_checked=True,all_inner_train_IDs_checked=True,independent_grid_selection=True,row_reversal=True,outer_label_poisoning=True,injected_leak_rejected=True,original_rank_ensemble_exact_reproduction=True,fixed_previous_two_seed_comparator=True,point_loss_only_plan=True,no_cross_arm_outer_selection=True,comparison_sha256=sha(O/'comparison.json')))
    save(O/'progress.json',dict(phase='complete',completed_outer=5));print(json.dumps(dict(primary=primary,models={k:v for k,v in models.items() if not k.startswith('raw_')},joint_improvement_gate_passed=bool(promotion)),ensure_ascii=True),flush=True)

if __name__=='__main__':main()
