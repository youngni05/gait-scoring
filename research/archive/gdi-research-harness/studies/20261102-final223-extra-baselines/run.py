"""Complete presentation baselines without changing earlier final evaluation files."""
from pathlib import Path
import json,sys,importlib.util,datetime,argparse
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'20261102-final223-frozen'
sys.path.insert(0,str(OLD))
import common as c
np=c.np;torch=c.torch;read=c.read;save=c.save;sha=c.sha
S=c.STUDY;SINGLE=S/'cnn-paper-input695';BIG=Path('D:/gait-research-data/D_P/final223-extra-baselines-20261102')
FOLLOW=HERE.parent/'20261102-final223-ranksim-followup'
LABELS={'single_CNN':'한 구간 CNN','multi_STT':'여러 구간 STT','multi_CNN':'여러 구간 CNN','CNN_bag':'CNN 확장 평균','fusion':'CNN·STT·좌우 보완 결합','plus_general':'기존 결합 + 일반 CNN','plus_ranksim':'기존 결합 + RankSim CNN'}

def check():
    f=read(HERE/'freeze.json');assert sha(__file__)==f['code_sha256'] and sha(HERE/'protocol.json')==f['protocol_sha256'] and sha(HERE/'models.json')==f['models_sha256']
    for p,h in f['protected'].items():assert sha(p)==h,('prior artifact changed',p)
    return read(HERE/'models.json')

def freeze():
    assert not (HERE/'freeze.json').exists();c.locked()
    reserved=set(read(OLD/'input-membership.json')['people']);members=[]
    for fold in range(5):
        folder=SINGLE/('received/results/fold%d'%fold);done=read(folder/'complete.json');fit=read(folder/'refit/fit_complete.json');curve=read(folder/'refit/curve.json')
        assert sha(folder/'refit/curve.json')==fit['curve_sha256'];assert done['checkpoint_sha256']==fit['checkpoint_sha256']
        ident=fit['identity'];assert ident==done['identity'] and not reserved&(set(ident['train_people'])|set(ident['validation_people']))
        assert ident['purpose']=='outer_refit' and ident['outer_fold']==fold
        assert curve[-1]['calibration']==done['calibration']
        members.append(dict(fold=fold,checkpoint=str(BIG/('checkpoints/fold%d.pt'%fold)),checkpoint_sha256=done['checkpoint_sha256'],remote_checkpoint='/abr/RESEARCH_USER/gdi-cnn-paper-input-20261030/runs/main/fold%d/refit/latest.pt'%fold,calibration=done['calibration'],replay_source=str(folder/'predictions.npz'),replay_sha256=sha(folder/'predictions.npz'),identity=ident))
    save(HERE/'models.json',dict(single_members=members,single_model_source=str(SINGLE/'bundle/model.py'),single_model_source_sha256=sha(SINGLE/'bundle/model.py')))
    protocol=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),request='Include one-window CNN, multiwindow STT and multiwindow CNN in presentation comparison',classification='supplemental descriptive evaluation requested after previous holdout results; not part of original three-model preregistration',people=222,records=377,cohort='Exactly existing377 scored records, no exclusions/label changes',single_CNN='Existing five single-window refit CNNs, original canonical124frame IDs per record, same full-record preprocessing, stored training-only OLS; equal average of five fold predictions',multi_STT='Reuse already locked per-record STT_f_refit predictions; equal average of five folds. Same multiwindow inputs as multi_CNN.',precision='CPU float32 with original models, replay absolute tolerance1e-4; no training or calibration fitting',quality='Retain previous input gate and cohort. Single canonical features must reproduce same source process; no new favorable-window selection.',statistics='Person-equal MAE, pooled record-side Pearson, person-equal bias, person bootstrapMAE95 (10000 draws seed20261105); fold-origin SD diagnostic only. No new superiority tests or winner selection.',presentation='Keep development692 OOF results separate. All final222 deployment rows average five fold-specific predictors; CNN_bag uses additional inner models. Flag late-added rows as supplemental.',scope='CP-source held-out people, not healthy-person or older-adult clinical validation')
    save(HERE/'protocol.json',protocol)
    protected={str(p.resolve()):sha(p) for folder in [OLD,FOLLOW] for p in folder.iterdir() if p.is_file()}
    protected.update({str((c.BIG/name).resolve()):sha(c.BIG/name) for name in ['features.npz','predictions.npz','scored.npz']})
    protected[str((SINGLE/'bundle/model.py').resolve())]=sha(SINGLE/'bundle/model.py')
    save(HERE/'freeze.json',dict(protocol_sha256=sha(HERE/'protocol.json'),models_sha256=sha(HERE/'models.json'),code_sha256=sha(__file__),protected=protected))
    print('ADDITIONAL_BASELINES_FROZEN',flush=True)

def prepare():
    check();prev=c.module('canonical_prepare',OLD/'prepare.py')
    with np.load(c.BIG/'features.npz') as z:videos=z['video_id'];ids=z['person_id'];oldx=z['x'];owner=z['window_record']
    rows_by_video={r['video_id']:r for r in read(OLD/'input-membership.json')['records']}
    rows=[rows_by_video[v] for v in videos];index={v:i for i,v in enumerate(videos)};values=np.empty((len(rows),2,124,12),dtype=np.float32);checks=[]
    def convert(row,raw,digest,ambiguous):
        xx,q,canonical=prev.convert(raw,np.array(row['frames']));i=index[row['video_id']]
        assert q['eligible'] and np.array_equal(xx,oldx[owner==i]),'Preprocessing or existing input changed'
        values[i]=canonical.astype(np.float32);checks.append(dict(video_id=row['video_id'],canonical_start=int(row['frames'][0]),canonical_end=int(row['frames'][-1]),multi_features_exact=True))
    prev.scan(rows,convert);assert np.isfinite(values).all()
    np.savez_compressed(BIG/'single-features.npz',x=values,person_id=ids,video_id=videos)
    save(HERE/'input-audit.json',dict(people=222,records=377,source_multi_features_exact=True,records_checked=checks,single_features_sha256=sha(BIG/'single-features.npz'),original_labels_reopened=False))
    print('CANONICAL_INPUTS_PREPARED',flush=True)

def predict():
    manifest=check();assert not (HERE/'evaluation-started.json').exists()
    model_source=c.module('single_cnn_model',manifest['single_model_source'])
    with np.load(BIG/'single-features.npz') as z:x=z['x'];ids=z['person_id'];videos=z['video_id']
    assert sha(BIG/'single-features.npz')==read(HERE/'input-audit.json')['single_features_sha256']
    with np.load(SINGLE/'bundle/data.npz') as z:devx=z['x']
    preds={};proofs=[]
    for m in manifest['single_members']:
        assert sha(m['checkpoint'])==m['checkpoint_sha256'];ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=True)
        assert ck['identity']==m['identity'];model=model_source.GdiCNN();model.load_state_dict(ck['model'],strict=True)
        with np.load(m['replay_source']) as z:ix=z['row_index'][:3];expected=z['predictions'][:3,0]
        a,b=m['calibration'];replay=c.prior.forward(model,'CNN',devx[ix],ck['lo'],ck['span'])*a+b
        error=float(abs(replay-expected).max());assert error<1e-4
        preds['single_%d'%m['fold']]=c.prior.forward(model,'CNN',x,ck['lo'],ck['span'])*a+b
        proofs.append(dict(fold=m['fold'],replay_max=error,checkpoint_sha256=m['checkpoint_sha256']))
    preds['single_CNN']=np.mean([preds['single_%d'%f] for f in range(5)],axis=0)
    with np.load(c.BIG/'predictions.npz') as z:
        for key,value in [('person_id',ids),('video_id',videos)]:np.testing.assert_array_equal(value,z[key])
        for f in range(5):preds['stt_%d'%f]=z['STT_%d_refit'%f]
    preds['multi_STT']=np.mean([preds['stt_%d'%f] for f in range(5)],axis=0)
    np.savez_compressed(BIG/'predictions.npz',person_id=ids,video_id=videos,**preds)
    save(HERE/'predictions-locked.json',dict(predictions_sha256=sha(BIG/'predictions.npz'),protocol_sha256=sha(HERE/'protocol.json'),model_replays=proofs,optimizer_steps=0,labels_reopened=False))
    check();print('BASELINE_PREDICTIONS_LOCKED',flush=True)

def evaluate():
    check();assert not (HERE/'evaluation-started.json').exists()
    assert sha(BIG/'predictions.npz')==read(HERE/'predictions-locked.json')['predictions_sha256']
    save(HERE/'evaluation-started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),post_hoc=True))
    with np.load(c.BIG/'scored.npz') as z:y=z['y'];ids=z['person_id'];videos=z['video_id'];p={'multi_CNN':z['CNN_refit'],'CNN_bag':z['CNN_bag'],'fusion':z['fusion']}
    with np.load(BIG/'predictions.npz') as z:
        np.testing.assert_array_equal(videos,z['video_id']);np.testing.assert_array_equal(ids,z['person_id'])
        p.update({k:z[k] for k in ['single_CNN','multi_STT']});fold_pred={k:z[k] for k in z.files if k.startswith('single_') or k.startswith('stt_')}
    with np.load(Path('D:/gait-research-data/D_P/final223-ranksim-followup-20261102/scored.npz')) as z:
        for key,value in [('person_id',ids),('video_id',videos),('y',y)]:np.testing.assert_array_equal(value,z[key])
        p.update({k:z[k] for k in ['plus_general','plus_ranksim']})
    people=np.unique(ids);draw=np.random.default_rng(20261105).integers(0,len(people),(10000,len(people)));metrics={}
    for name in LABELS:
        pp=p[name];err=c.person_errors(y,pp,ids)
        metrics[name]=dict(label=LABELS[name],mae=float(err.mean()),mae95=np.percentile(err[draw].mean(1),[2.5,97.5]).tolist(),pearson=float(np.corrcoef(y.ravel(),pp.ravel())[0,1]),bias=float(np.mean([(pp[ids==i]-y[ids==i]).mean() for i in people])))
    foldstats={}
    for name,pattern in [('single_CNN','single_%d'),('multi_STT','stt_%d')]:
        vals=[float(c.person_errors(y,fold_pred[pattern%f],ids).mean()) for f in range(5)]
        foldstats[name]=dict(mae_per_origin_fold=vals,sd=float(np.std(vals,ddof=1)),not_five_independent_tests=True)
    np.savez_compressed(BIG/'scored.npz',y=y,person_id=ids,video_id=videos,**p)
    save(HERE/'results.json',dict(people=222,records=377,metrics=metrics,fold_diagnostics=foldstats,post_hoc_supplement=True,previous_evaluations_unchanged=True,scored_sha256=sha(BIG/'scored.npz'),no_training_or_retuning=True))
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

def verify():
    manifest=check();r=read(HERE/'results.json');assert sha(BIG/'scored.npz')==r['scored_sha256']
    with np.load(BIG/'scored.npz') as z:y=z['y'];ids=z['person_id'];p={k:z[k] for k in LABELS}
    with np.load(BIG/'predictions.npz') as z:
        for name,pattern in [('single_CNN','single_%d'),('multi_STT','stt_%d')]:np.testing.assert_allclose(sum(z[pattern%f] for f in range(5))/5,p[name],atol=1e-12,rtol=0)
    for name,pp in p.items():
        errors=[]
        for person in sorted(set(ids)):
            rows=[i for i,v in enumerate(ids) if v==person]
            errors.append(sum(abs(float(pp[i,s])-float(y[i,s])) for i in rows for s in [0,1])/(len(rows)*2))
        assert abs(sum(errors)/len(errors)-r['metrics'][name]['mae'])<1e-12
        rng=np.random.default_rng(20261105);dist=[]
        for _ in range(10000):dist.append(np.mean([errors[i] for i in rng.integers(0,len(errors),len(errors))]))
        np.testing.assert_allclose(np.percentile(dist,[2.5,97.5]),r['metrics'][name]['mae95'],atol=1e-12,rtol=0)
    save(HERE/'verification.json',dict(passed=True,original_artifacts_unchanged=True,independent_person_mae=True,independent_mae_intervals=True,fold_averages_verified=True,model_replay_count=5,results_sha256=sha(HERE/'results.json')))
    print('EXTRA_BASELINES_VERIFIED')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','predict','evaluate','verify']);a=ap.parse_args();globals()[a.stage]()
