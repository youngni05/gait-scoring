"""Resume the frozen experiment; preserve completed models and original code."""
from pathlib import Path
import json, os, shutil, sys, warnings, datetime
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e
import numpy as np
import joblib
from sklearn.base import clone
import subprocess
warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMRegressor was fitted with feature names')
HERE=e.HERE;HEAVY=e.HEAVY

def main():
    prep=json.loads((HERE/'preparation.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text())
    assert e.sha(HEAVY/'features.npz')==prep['features_sha256']
    assert e.sha(HERE/'protocol.json')==prep['protocol_sha256']
    assert e.sha(HERE/'experiment.py')==protocol['code_sha256']
    assert e.sha(HERE/'effective-splits.json')==prep['effective_splits_sha256']
    d=np.load(HEAVY/'features.npz');X=d['X'];y=d['y'];ids=d['ids']
    schema=json.loads((HERE/'feature-schema.json').read_text())
    splits=json.loads((HERE/'effective-splits.json').read_text());pool=e.candidates()
    checkpoint=HERE/'resume-checkpoint.json'
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text());out=state['selection'];pred=state['predictions']
    else:
        out=json.loads((HERE/'selection.json').read_text());pred=json.loads((HEAVY/'predictions.json').read_text())
    done={(s['repeat'],s['fold'],s['view']) for s in out};assert len(done)==len(out)
    for s in out:
        stem=f"r{s['repeat']}-f{s['fold']}-{s['view']}";path=HEAVY/'models'/(stem+'.joblib')
        assert e.sha(path)==s['model_sha256']
        rec=[r for r in pred if (r['repeat'],r['fold'],r['view'])==(s['repeat'],s['fold'],s['view'])]
        assert [r['index'] for r in rec]==s['test']
        xx=X[:,schema['views'][s['view']]];pp=joblib.load(path).predict(xx[s['test']])
        np.testing.assert_allclose(pp,[r['pred'] for r in rec],atol=1e-9,rtol=0)
    original_hashes={f"r{s['repeat']}-f{s['fold']}-{s['view']}.joblib":s['model_sha256'] for s in out}
    resumed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    e.save(HERE/'resume-state.json',{'status':'running','resumed_at':resumed_at,'original_completed':len(out),
      'preserved_model_hashes':original_hashes,'protocol_sha256':e.sha(HERE/'protocol.json'),'resume_code_sha256':e.sha(__file__)})
    print('RESUMED_PRESERVED',len(out),'MODELS',flush=True)
    for repeat in splits:
      rr=repeat['repeat']
      for fold in repeat['splits']:
        ff=fold['fold'];tr=np.array(fold['train']);te=np.array(fold['test'])
        assert not set(ids[tr])&set(ids[te]);assert len(set(ids))==len(ids)
        for view in e.VIEWS:
          if (rr,ff,view) in done:continue
          xx=X[:,schema['views'][view]];losses={};innerpred={}
          for name,template in pool.items():
            pp=np.full(len(y),np.nan)
            for inner in fold['inner']:
              it=np.array(inner['train']);iv=np.array(inner['validation'])
              assert set(it)<=set(tr) and set(iv)<=set(tr) and not set(it)&set(iv)
              m=clone(template).fit(xx[it],y[it]);pp[iv]=m.predict(xx[iv])
            assert np.isfinite(pp[tr]).all()
            losses[name]=float(np.mean(np.abs(pp[tr]-y[tr])));innerpred[name]=pp[tr]
          best=min(losses,key=losses.get);model=clone(pool[best]).fit(xx[tr],y[tr]);pp=model.predict(xx[te])
          stem=f'r{rr}-f{ff}-{view}';path=HEAVY/'models'/(stem+'.joblib')
          for p in [path,HEAVY/'models'/(stem+'-inner.npz')]:
            if p.exists():
              backup=HEAVY/'interrupted-unrecorded'/p.name;backup.parent.mkdir(exist_ok=True)
              assert not backup.exists();shutil.copy2(p,backup)
          joblib.dump(model,path);np.savez_compressed(HEAVY/'models'/(stem+'-inner.npz'),**innerpred)
          out.append({'repeat':rr,'fold':ff,'view':view,'selected':best,'inner_MAE':losses,
            'model_sha256':e.sha(path),'train':tr.tolist(),'test':te.tolist()})
          for i,p in zip(te,pp):pred.append({'repeat':rr,'fold':ff,'view':view,'index':int(i),'id':str(ids[i]),'true':float(y[i]),'pred':float(p)})
          done.add((rr,ff,view))
          temp=checkpoint.with_suffix('.tmp');e.save(temp,{'selection':out,'predictions':pred});os.replace(temp,checkpoint)
          print('COMPLETED',len(out),'/100',rr,ff,view,flush=True)
        e.save(HERE/'selection.json',out);e.save(HEAVY/'predictions.json',pred)
    assert len(out)==100
    for name,digest in original_hashes.items():assert e.sha(HEAVY/'models'/name)==digest
    subprocess.run([sys.executable,'-W','ignore::UserWarning',str(HERE/'verify.py')],check=True)
    state=json.loads((HERE/'resume-state.json').read_text());state.update(status='verified_complete',completed_models=len(out),verification_sha256=e.sha(HERE/'verification.json'))
    e.save(HERE/'resume-state.json',state)
    print('RESUMED_STUDY_VERIFIED',flush=True)
if __name__=='__main__':main()
