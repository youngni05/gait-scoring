"""Freeze existing checkpoints and input identities; no inference/training."""
from pathlib import Path
import json,hashlib,datetime
ROOT=Path('D:/Projects/D_P');H=Path(__file__).resolve().parent;S=H.parent;T=S/'20261030-gpu-motionbert-study';BIG=Path('D:/gait-research-data/D_P/gdi-case-audit-20261105')
def resolve(p):
 s=str(p).replace('\\','/');pre='C:/Users/jun/Desktop/D_P/'
 return Path('D:/Projects/D_P/'+s[len(pre):]) if s.startswith(pre) else Path(s)
def read(p):return json.loads(resolve(p).read_text(encoding='utf-8-sig'))
def sha(p):
 v=hashlib.sha256()
 with resolve(p).open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):v.update(chunk)
 return v.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def prepare():
 assert not (H/'freeze.json').exists()
 members=[];protected={};reserved=set(read(S/'20261102-final223-frozen/input-membership.json')['people']);weights={}
 def protect(p,expected=None):
  p=resolve(p);v=sha(p)
  if expected:assert v==expected,str(p)
  protected[str(p)]=v;return p
 for folder in ['20261102-final223-frozen','20261102-final223-ranksim-followup','20261102-final223-extra-baselines']:
  h=S/folder;r=read(h/'results.json');q=read(h/'verification.json');assert q['passed'];protect(h/'results.json',q['results_sha256'])
  for n in ['protocol.json','models.json','verification.json','freeze.json','predictions-locked.json']:protect(h/n)
 for folder in ['final223-frozen-20261102','final223-ranksim-followup-20261102','final223-extra-baselines-20261102']:
  for n in ['predictions.npz','scored.npz']:protect(Path('D:/gait-research-data/D_P')/folder/n)
 for folder,n in [('final223-frozen-20261102','features.npz'),('final223-extra-baselines-20261102','single-features.npz')]:protect(Path('D:/gait-research-data/D_P')/folder/n)
 prior=protect(S/'20261101-self-video-gdi/predict.py')
 for source in [T/'cnn-multiwindow695/bundle/model.py',T/'stt692/bilateral/bundle/bilateral.py']:
  protect(source)
 for source in (T/'stt692/multiwindow/bundle/vendor').rglob('*.py'):protect(source)
 base=read(S/'20261102-final223-frozen/models.json')
 for m in base['members']:
  f=m['fold'];family=m['family'];key='base_bilateral_%d'%f if family=='bilateral' else 'base_%s_%d_%s'%(family,f,m['tag'])
  cp=protect(m['checkpoint'],m['checkpoint_sha256']);members.append(dict(key=key,role=family,fold=f,tag=m.get('tag'),family=family,precision='float32',checkpoint=str(cp),checkpoint_sha256=sha(cp),calibration=m['calibration'],model_source=str(prior),model_source_sha256=sha(prior),replay_kind='old_final_member',replay_key=key.removeprefix('base_')))
 singles=read(S/'20261102-final223-extra-baselines/models.json');src=protect(singles['single_model_source'],singles['single_model_source_sha256'])
 for m in singles['single_members']:
  assert not reserved&(set(m['identity']['train_people'])|set(m['identity']['validation_people']))
  cp=protect(m['checkpoint'],m['checkpoint_sha256']);members.append(dict(key='single_%d'%m['fold'],role='single',fold=m['fold'],family='CNN',precision='float32',checkpoint=str(cp),checkpoint_sha256=sha(cp),calibration=m['calibration'],model_source=str(src),model_source_sha256=sha(src),replay_kind='old_single',replay_key='single_%d'%m['fold']))
 old=read(S/'20261102-final223-ranksim-followup/models.json')
 for m in old['members']:
  role='general' if m['arm']=='scratch_control' else 'Rank1';f=m['fold'];assert m['reserved_overlap']==0
  cp=protect(m['checkpoint'],m['checkpoint_sha256']);src=protect(m['model_source'],m['model_source_sha256'])
  members.append(dict(key='%s_%d'%(role,f),role=role,fold=f,family='CNN',precision='float64',checkpoint=str(cp),checkpoint_sha256=sha(cp),calibration=m['calibration'],model_source=str(src),model_source_sha256=sha(src),replay_kind='old_added',replay_key='%s_%d'%(m['arm'],f)))
 weights['general']=old['weights']['scratch_control'];weights['Rank1']=old['weights']['ranksim']
 reg=read(ROOT/'gdi-research-harness/control/gpu-watch/registry.json')
 for study,arms in [('cnn-ranksim692',{'mse':'MSE'}),('cnn-huber-point692',{'mse':'HuberOnly','ranksim':'HuberRank1'}),('cnn-mae-point692-v2',{'mse':'MAEOnly','ranksim':'MAERank1'})]:
  h=T/study;rid={'cnn-ranksim692':'cnn-ranksim-692','cnn-huber-point692':'cnn-huber-point-692','cnn-mae-point692-v2':'cnn-mae-point-692'}[study];row=next(e for e in reg['experiments'] if e['id']==rid);assert row['review_state']=='reported'
  report=protect(h/('analysis/comparison.json' if study=='cnn-ranksim692' else 'analysis/combined-comparison.json'),row['reported_digest']);q=read(report);assert q['qualified_full_verification_passed']
  receipt=read(h/('checks/closeout-receipt.json' if study=='cnn-ranksim692' else 'checks/full-receipt.json'));u=resolve(receipt['archive']).parent/'performance-review-originals';protect(u/'manifest.json',row['manifest_sha256']);src=protect(h/'model.py')
  if study!='cnn-ranksim692':
   select=protect(h/'ensemble/analysis/selection.json');selected=read(select)['folds']
   for role in arms.values():weights[role]=[next(x['additional_weight'] for x in selected if x['arm']==role and x['outer']==f) for f in range(5)]
  for arm,role in arms.items():
   for f in range(5):
    folder=u/'runs/main'/arm/('fold%d'%f);done=read(protect(folder/'complete.json'));ident=read(protect(folder/'refit/identity.json'));assert not reserved&(set(ident['train_people'])|set(ident['validation_people'])) and ident['purpose']=='outer_refit' and ident['outer_fold']==f
    cp=protect(folder/'refit/latest.pt',done['checkpoint_sha256']);replay=protect(folder/'predictions.npz',done['prediction_sha256']);members.append(dict(key='%s_%d'%(role,f),role=role,fold=f,family='CNN',precision='float64',checkpoint=str(cp),checkpoint_sha256=sha(cp),calibration=done['calibration'],model_source=str(src),model_source_sha256=sha(src),replay_kind='dev_multi',replay_source=str(replay),replay_sha256=sha(replay)))
 protect(T/'cnn-record-loss692/bundle/data.npz');protect(S/'20261102-final223-frozen/input-membership.json');assert len(members)==94 and len({m['key'] for m in members})==94
 save(H/'models.json',dict(members=members,selections=base['selections'],addition_weights=weights,reserved_people_count=len(reserved),protected=protected,prior_source=str(prior),prior_source_sha256=sha(prior)))
 # Seal code only after all runner files are present and syntax checked.
 code={p.name:sha(p) for p in H.glob('*.py')};save(H/'freeze.json',dict(protocol_sha256=sha(H/'protocol.json'),models_sha256=sha(H/'models.json'),code=code,created_at=datetime.datetime.now().astimezone().isoformat(),new_training=False,prior_outcomes_already_known=True))
 print(json.dumps(dict(frozen=True,checkpoints=len(members),protocol_sha256=sha(H/'protocol.json'),models_sha256=sha(H/'models.json'))),flush=True)
if __name__=='__main__':prepare()
