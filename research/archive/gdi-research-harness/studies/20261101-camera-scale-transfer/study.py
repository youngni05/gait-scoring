"""Single-parameter speed scale transfer, held-out person and opposite camera."""
from pathlib import Path
import sys,json,hashlib,datetime,argparse,time,os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
REFERENCE=HERE.parent/'20261101-camera-height-speed'
STORE=Path('D:/gait-research-data/D_P/camera-scale-transfer-20261101')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def source():
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for p,h in protocol['sources'].items():assert sha(p)==h
    d=pd.read_csv(REFERENCE/'main-clips.csv');assert len(d)==56 and d.subject.nunique()==14 and not d.duplicated(['subject','camera','condition']).any()
    return d
def tasks(mode,d):
    people=sorted(d.subject.unique().tolist());people=people if mode=='main' else people[:2]
    return [(int(sid),camera) for sid in people for camera in ['PG1','PG2']]
def work(task,d):
    sid,cam=task;tag=f's{sid:02d}-train{cam}';ledger=STORE/f'{tag}.json'
    if ledger.exists():
        q=json.loads(ledger.read_text());assert q['code_sha256']==sha(__file__) and q['prediction_sha256']==sha(q['prediction_path']);return q
    train=d[d.subject.ne(sid)&d.condition.eq('straight')&d.camera.eq(cam)].sort_values('subject')
    test=d[d.subject.eq(sid)].sort_values(['condition','camera']).copy()
    assert len(train)==13 and len(test)==4 and set(train.subject).isdisjoint(test.subject)
    x=train.pinhole_ols.to_numpy();y=train.truth_ols.to_numpy();alpha=float(np.dot(x,y)/np.dot(x,x))
    assert np.isfinite(alpha) and alpha>0
    out=test[['key','subject','camera','condition','truth_ols','pinhole_ols','plane_ols']].copy()
    out['training_camera']=cam;out['transfer']=np.where(out.camera.eq(cam),'same_camera','opposite_camera');out['alpha']=alpha
    out['RawPinhole']=out.pinhole_ols;out['ScaledPinhole']=out.pinhole_ols*alpha
    path=STORE/f'{tag}.csv';out.to_csv(path,index=False)
    q=dict(task=list(task),training_people=train.subject.tolist(),training_keys=train.key.tolist(),training_condition='straight',alpha=alpha,model_path=str(STORE/f'{tag}-scale.json'),prediction_path=str(path),prediction_sha256=sha(path),code_sha256=sha(__file__))
    save(q['model_path'],dict(alpha=alpha,training_people=q['training_people'],training_camera=cam,held_out_subject=sid));q['model_sha256']=sha(q['model_path']);save(ledger,q);return q
def prepare():
    paths=[REFERENCE/n for n in ['study.py','verify.py','protocol.json','main-verification.json','main-clips.csv','metrics.csv','paired.csv','README.md']]
    save(HERE/'protocol.json',dict(code_sha256=sha(__file__),sources={str(p):sha(p) for p in paths},
      hypothesis='Pinhole height-based speed retains approximately0.10m/s positive bias after reducing camera differences. Test whether a common multiplicative scale learned from other people in one camera transfers to an unseen person in the other camera.',
      inputs='Same existing14people56clips2sec straight/turn,RGB-only MediaPipe,height,extra real camera intrinsics. No new image model. Reference speeds in13trainingpeople used to fit one scalar; held-outperson references evaluation only.',
      label='Provider horizontal hip-midpoint OLS translation speed,not Mobilise-D mean stride speed or SPPB/TUG.',
      models='RawPinhole frozen previous estimator; ScaledPinhole alpha*x. No offset,grid,feature/epoch/threshold search. Alpha=sum(x*y)/sum(x*x),each trainingperson one straight clip in one trainingcamera. Sameview and oppositeview evaluated separately.',
      split='14 leave-one-person-out folds,each trained separately on PG1 or PG2. All4clips of held-outperson excluded from alpha fit. Training13distinctpeople; no turn labels in fit. Predict sameperson4clips from both28scalar models =>112rows. Oppositecamera is within same recording system,not new phone/hospital domain.',
      seeds='Deterministic fit; no new5seed repetition. Existing Mobilise-D people/bouts/5seeds unchanged.14people reused exploratory development data,not independent external clinical validation.',
      selection='No hyperparameters or model selection. Same/opposite camera and straight/turn reported all,primary is opposite-camera straight. Each testperson has two oppositecamera predictions,person-equal.',
      success='Before this new experiment: opposite-camera straight MAE decreases>=0.02m/s vsRawPinhole; both per-targetcamera MAE<=0.10m/s; corrected paired-camera mean difference<=0.10m/s. Engineering targets,not clinical standards. Report each component and failures,not retune.',
      expected=dict(pilot_tasks=4,pilot_predictions=16,main_tasks=28,main_predictions=112,people=14,scalar_models=28,metric_rows=8,camera_metric_rows=16,paired_rows=8),
      verification='Source hashes,person/clip links,28 savedalpha replay and independent lstsq refit,112predictions recalculated,tolerance1e-12. Independent MAE/bias/paired recalculation. Pilot first2heldpeople4tasks then main.',
      resource='CPU1,noGPU,estimate1-3min,scalar models/predictions<1MB D;smallcode/protocol/summary C.',
      limitations='Shared scale can correct more than headtop/span geometry and does not identify cause. Requires research reference-speed calibration in trainingpeople plus actual intrinsic calibration; not height-only ready phone estimator.2sec and14healthy-sourcepeople do not establish gait scores or clinical accuracy.',
      storage=dict(small=str(HERE),large=str(STORE))))
def run(mode):
    started=time.time();d=source();STORE.mkdir(parents=True,exist_ok=True);ts=tasks(mode,d)
    for i,t in enumerate(ts,1):work(t,d);print('FIT_SCALE',i,t,flush=True)
    save(HERE/f'{mode}-completion.json',dict(tasks=len(ts),predictions=4*len(ts),people=len(ts)//2,scalar_models=len(ts),code_sha256=sha(__file__),seconds=time.time()-started))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode)
