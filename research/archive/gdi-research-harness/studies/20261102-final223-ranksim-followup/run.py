"""Post-hoc fixed-weight follow-up; preserves the completed primary evaluation."""
from pathlib import Path
import sys,json,hashlib,importlib.util,datetime,argparse
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'20261102-final223-frozen'
spec=importlib.util.spec_from_file_location('original_common',OLD/'common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
np=c.np;torch=c.torch;read=c.read;save=c.save;sha=c.sha
S=c.STUDY;BIG=Path('D:/gait-research-data/D_P/final223-ranksim-followup-20261102')
E=S/'ranksim-ensemble692';NAMES=['CNN_bag','fusion','plus_general','plus_ranksim']

def check():
    f=read(HERE/'freeze.json');assert sha(__file__)==f['code_sha256']
    assert sha(HERE/'protocol.json')==f['protocol_sha256'] and sha(HERE/'models.json')==f['models_sha256']
    for path,digest in f['original_files'].items():assert sha(path)==digest,('original artifact changed',path)
    for path,digest in f['original_large'].items():assert sha(path)==digest
    for path,digest in f['external_code'].items():assert sha(path)==digest
    return read(HERE/'models.json')

def freeze():
    assert not (HERE/'freeze.json').exists(),'Never overwrite this follow-up freeze'
    c.locked();verification=read(OLD/'verification.json');assert verification['passed']
    assert sha(OLD/'results.json')==verification['results_sha256']
    sel=read(E/'analysis/selection.json');report=read(E/'analysis/comparison.json')
    assert sha(E/'analysis/selection.json')==report['selection_sha256']
    assert sha(E/'analysis/comparison.json')==read(E/'analysis/verification.json')['comparison_sha256']
    weights={arm:[next(r['additional_weight'] for r in sel['folds'] if r['arm']==arm and r['outer']==f) for f in range(5)] for arm in ['scratch_control','ranksim']}
    assert weights=={'scratch_control':[.35,.2,.2,.1,.2],'ranksim':[.45,.05,.15,.35,.3]}
    reserved=set(read(OLD/'input-membership.json')['people']);members=[]
    for arm,study in [('scratch_control',c.H/'studies/20261101-synthgait-conv-transfer'),('ranksim',S/'cnn-ranksim692')]:
        root=Path(read(study/'checks/closeout-receipt.json')['archive']).parent/'performance-review-originals'
        model_source=root/'model.py'
        for f in range(5):
            folder=root/'runs/main'/('scratch' if arm=='scratch_control' else 'ranksim')/('fold%d'%f)
            ckpath=folder/'refit/latest.pt';done=read(folder/'refit/fit_complete.json');curve=read(folder/'refit/curve.json')
            complete=read(folder/'complete.json')
            assert sha(folder/'complete.json')==sel['source_hashes'][str(folder/'complete.json')]
            assert sha(ckpath)==done['checkpoint_sha256']==complete['checkpoint_sha256']
            assert sha(folder/'refit/curve.json')==done['curve_sha256']
            ck=torch.load(ckpath,map_location='cpu',weights_only=True);ident=ck['identity'];assert ident==done['identity']
            assert not reserved&(set(map(str,ident['train_people']))|set(map(str,ident['validation_people'])))
            assert ident['outer_fold']==f and ident['purpose']=='outer_refit'
            members.append(dict(arm=arm,fold=f,checkpoint=str(ckpath),checkpoint_sha256=sha(ckpath),model_source=str(model_source),model_source_sha256=sha(model_source),calibration=curve[-1]['calibration'],additional_weight=weights[arm][f],replay_source=str(folder/'predictions.npz'),replay_source_sha256=sha(folder/'predictions.npz'),train_people=len(ident['train_people']),validation_people=len(ident['validation_people']),reserved_overlap=0))
    save(HERE/'models.json',dict(members=members,weights=weights,selection_source=str(E/'analysis/selection.json'),selection_sha256=sha(E/'analysis/selection.json')))
    protocol=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User authorized supplemental fixed-weight general-CNN and RankSim additions on the same222 people/377records; preserve completed original evaluation.',classification='post-hoc supplemental comparison after original holdout outcomes were observed; not a new untouched final test',cohort='Exact original scored222 people/377records, same row order and outcomes; no new exclusions or samples',baseline='Reuse original frozen CNN_bag and fusion outputs and each fusion_f; do not recalibrate or recompute them',added_models='Five pre-existing refit CNN checkpoints per arm, native12features124frames, window mean then each existing training-only calibration; no training',weights=weights,formula='For each f: (1-w[f])*original_fusion_f + w[f]*added_refit_f; then equal average across five f. Never combine global averages using a mean weight.',precision='Additional CNNs CPU float64 to match their previously qualified replay path; original fixed float32 predictions unchanged. Require every new model development replay max absolute error<1e-4.',primary_metric='Mean absolute error across record-sides within person, then equal mean across222 people',primary_comparisons=['plus_general minus CNN_bag','plus_ranksim minus CNN_bag'],bootstrap=dict(draws=10000,seed=20261103,unit='paired person, fixed prediction errors',individual95=[2.5,97.5],familywise95_Bonferroni_two_comparisons=[1.25,98.75]),decision='For each primary comparison, evidence favoring added model only if observed difference<0 and upper Bonferroni97.5 interval<0; otherwise inconclusive unless lower interval>0. Report individual95 too; not equivalence testing.',secondary=['each addition vs original fusion, descriptive95','RankSim addition vs general addition, descriptive95','per-model MAE95/Pearson/bias','original true-person-meanGDI>=90 subgroup count/MAE','five-origin-fold diagnostic MAEs and SD'],one_pass='Lock new predictions before re-reading original scored outcomes. Do not retune any weights or select strength/epoch after outcomes. Preserve all original artifacts byte-for-byte.',scope='same-source post-hoc supplemental comparison; no older-adult or external capture validity claim',storage=dict(small=str(HERE.resolve()),large=str(BIG)))
    save(HERE/'protocol.json',protocol)
    files={str(p.resolve()):sha(p) for p in OLD.iterdir() if p.is_file()}
    large={str((c.BIG/name).resolve()):sha(c.BIG/name) for name in ['features.npz','predictions.npz','scored.npz']}
    ext={str((OLD/'common.py').resolve()):sha(OLD/'common.py')}
    ext.update({m['model_source']:m['model_source_sha256'] for m in members})
    save(HERE/'freeze.json',dict(code_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),models_sha256=sha(HERE/'models.json'),original_files=files,original_large=large,external_code=ext,new_predictions_computed=False,prior_holdout_results_already_observed=True))
    print(json.dumps(dict(frozen=True,additional_checkpoints=len(members),weights=weights,classification=protocol['classification'])))

def forward(m,ck,x,model_source):
    parts=[];m.eval()
    with torch.inference_mode():
        for start in range(0,len(x),128):
            v=model_source.channels(torch.from_numpy(x[start:start+128])).reshape(-1,12,124).double()
            p=m(v)[:,0]*float(ck['span'])+float(ck['lo']);parts.append(p.reshape(-1,2).numpy())
    return np.concatenate(parts)

def predict():
    manifest=check();assert not (HERE/'evaluation-started.json').exists()
    with np.load(c.BIG/'features.npz') as z:x=z['x'];owner=z['window_record'];ids=z['person_id'];videos=z['video_id']
    with np.load(c.BIG/'predictions.npz') as z:out={k:z[k].copy() for k in ['fusion']+['fusion_%d'%f for f in range(5)]}
    with np.load(S/'cnn-multiwindow695/bundle/data.npz') as z:devx=z['x'];devowner=z['window_record'];devfold=z['fold']
    proofs=[]
    for member in manifest['members']:
        arm=member['arm'];f=member['fold'];ck=torch.load(member['checkpoint'],map_location='cpu',weights_only=True)
        assert sha(member['checkpoint'])==member['checkpoint_sha256']
        model_source=c.module('extra_model_%s_%d'%(arm,f),member['model_source']);model=model_source.GdiCNN().double();model.load_state_dict(ck['model'],strict=True)
        a,b=member['calibration']
        with np.load(member['replay_source']) as z:
            ix=z['row_index'][:2];expected=z['predictions'][:2,0];np.testing.assert_allclose(z['calibration'],[a,b],atol=0,rtol=0)
        replay=[]
        for i in ix:
            assert devfold[i]==f
            replay.append(forward(model,ck,devx[devowner==i],model_source).mean(0)*a+b)
        error=float(np.abs(np.array(replay)-expected).max());assert error<1e-4,(arm,f,error)
        raw=forward(model,ck,x,model_source);pred=c.aggregate(raw,owner,len(ids))*a+b
        key='%s_%d'%(arm,f);out[key]=pred;w=member['additional_weight']
        out['plus_%s_%d'%(arm,f)]=(1-w)*out['fusion_%d'%f]+w*pred
        proofs.append(dict(arm=arm,fold=f,checkpoint_sha256=member['checkpoint_sha256'],development_replay_max=error))
        print(json.dumps(dict(arm=arm,fold=f,replay_max=error)),flush=True)
    for name,arm in [('plus_general','scratch_control'),('plus_ranksim','ranksim')]:out[name]=np.mean([out['plus_%s_%d'%(arm,f)] for f in range(5)],axis=0)
    assert len(ids)==377 and len(np.unique(ids))==222 and all(np.isfinite(v).all() for v in out.values())
    np.savez_compressed(BIG/'predictions.npz',person_id=ids,video_id=videos,**out)
    save(HERE/'predictions-locked.json',dict(protocol_sha256=sha(HERE/'protocol.json'),predictions_sha256=sha(BIG/'predictions.npz'),model_replays=proofs,optimizer_steps=0,original_labels_reopened_in_followup=False))
    check();print('SUPPLEMENTAL_PREDICTIONS_LOCKED',flush=True)

def evaluate():
    check();lock=read(HERE/'predictions-locked.json');assert sha(BIG/'predictions.npz')==lock['predictions_sha256']
    assert not (HERE/'evaluation-started.json').exists()
    save(HERE/'evaluation-started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),post_hoc=True))
    with np.load(c.BIG/'scored.npz') as z:y=z['y'];ids=z['person_id'];videos=z['video_id'];p={k:z[k] for k in ['CNN_bag','fusion']}
    with np.load(BIG/'predictions.npz') as z:
        np.testing.assert_array_equal(ids,z['person_id']);np.testing.assert_array_equal(videos,z['video_id'])
        for name in ['plus_general','plus_ranksim']:p[name]=z[name]
        np.testing.assert_array_equal(p['fusion'],z['fusion'])
    errors={k:c.person_errors(y,v,ids) for k,v in p.items()};people=np.unique(ids)
    cfg=read(HERE/'protocol.json')['bootstrap'];draw=np.random.default_rng(cfg['seed']).integers(0,len(people),(cfg['draws'],len(people)))
    high=np.array([y[ids==i].mean()>=90 for i in people]);metrics={}
    for name,pred in p.items():
        e=errors[name];metrics[name]=dict(mae=float(e.mean()),mae95=np.percentile(e[draw].mean(1),[2.5,97.5]).tolist(),pearson=float(np.corrcoef(y.ravel(),pred.ravel())[0,1]),person_equal_bias=float(np.mean([(pred[ids==i]-y[ids==i]).mean() for i in people])),high90_people=int(high.sum()),high90_mae=float(e[high].mean()))
    comparisons={}
    for new,reference,primary in [('plus_general','CNN_bag',True),('plus_ranksim','CNN_bag',True),('plus_general','fusion',False),('plus_ranksim','fusion',False),('plus_ranksim','plus_general',False)]:
        delta=errors[new]-errors[reference];dist=delta[draw].mean(1);ci=np.percentile(dist,[2.5,97.5]);family=np.percentile(dist,[1.25,98.75]) if primary else None
        used=family if primary else ci;mean=float(delta.mean());verdict='lower_error_supported' if mean<0 and used[1]<0 else 'higher_error_supported' if used[0]>0 else 'inconclusive'
        comparisons[new+'_minus_'+reference]=dict(primary=primary,mae_delta=mean,reduction_percent=-100*mean/errors[reference].mean(),paired_person95=ci.tolist(),familywise95_Bonferroni_interval=family.tolist() if primary else None,verdict=verdict,people_improved=int((delta<-1e-12).sum()),people_worsened=int((delta>1e-12).sum()))
    result=dict(post_hoc_supplement=True,people=len(people),records=len(ids),metrics=metrics,comparisons=comparisons,protocol_sha256=sha(HERE/'protocol.json'),predictions_sha256=lock['predictions_sha256'],old_primary_results_preserved=True,no_training_or_retuning=True,bootstrap=cfg)
    np.savez_compressed(BIG/'scored.npz',y=y,person_id=ids,video_id=videos,**p)
    result['scored_sha256']=sha(BIG/'scored.npz');save(HERE/'results.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

def verify():
    manifest=check();r=read(HERE/'results.json');assert sha(BIG/'scored.npz')==r['scored_sha256']
    with np.load(BIG/'predictions.npz') as z:d={k:z[k] for k in z.files}
    with np.load(BIG/'scored.npz') as z:y=z['y'];ids=z['person_id'];p={k:z[k] for k in NAMES}
    rebuild_error=0.
    for name,arm in [('plus_general','scratch_control'),('plus_ranksim','ranksim')]:
        rebuilt=sum(d['fusion_%d'%f]+manifest['weights'][arm][f]*(d['%s_%d'%(arm,f)]-d['fusion_%d'%f]) for f in range(5))/5
        e=float(np.abs(rebuilt-p[name]).max());assert e<1e-10;rebuild_error=max(rebuild_error,e)
    pe={k:[] for k in NAMES}
    for person in sorted(set(ids)):
        rows=[i for i,v in enumerate(ids) if v==person]
        for name in NAMES:pe[name].append(sum(abs(float(p[name][i,s])-float(y[i,s])) for i in rows for s in [0,1])/(2*len(rows)))
    for name,v in pe.items():assert abs(sum(v)/len(v)-r['metrics'][name]['mae'])<1e-12
    cfg=r['bootstrap'];rng=np.random.default_rng(cfg['seed']);distributions={k:[] for k in ['plus_general','plus_ranksim']}
    for _ in range(cfg['draws']):
        ix=rng.integers(0,len(pe['CNN_bag']),len(pe['CNN_bag']))
        for name in distributions:distributions[name].append(sum(pe[name][i]-pe['CNN_bag'][i] for i in ix)/len(ix))
    for name,v in distributions.items():
        comp=r['comparisons'][name+'_minus_CNN_bag']
        np.testing.assert_allclose(np.percentile(v,[2.5,97.5]),comp['paired_person95'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.percentile(v,[1.25,98.75]),comp['familywise95_Bonferroni_interval'],atol=1e-12,rtol=0)
    out=dict(passed=True,results_sha256=sha(HERE/'results.json'),original_artifacts_unchanged=True,independent_mae=True,independent_pair_bootstrap=True,independent_familywise_intervals=True,independent_blend_max_difference=rebuild_error,additional_models=10,people=222,records=377,post_hoc=True)
    save(HERE/'verification.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','predict','evaluate','verify']);a=ap.parse_args();globals()[a.stage]()
