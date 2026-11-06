"""Check split/sample/weight invariants before any sensitivity fits."""
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];B=HERE/'bundle';OLD=ROOT/'studies/20261105-gdi-masked-pretrain-v3/bundle'
sys.path[:0]=[str(B),str(ROOT/'outputs/motionbert-smoke-20261015/deps')]
import experiment as e
p=e.read(B/'protocol.json');d=e.load_data(Path('D:/gait-research-data/D_P/gdi-masked-pretrain-v3-20261105/data.npz'),p)
for file in ['cnn.py','CNN-original.json','STT-original.json']:
 assert e.sha(B/file)==e.sha(OLD/file)
for file in (OLD/'model').rglob('*.py'):assert e.sha(file)==e.sha(B/file.relative_to(OLD))
# Functional change boundary: restore the three controlled additions and compare ASTs.
s=(B/'experiment.py').read_text()
s=s.replace("variant=protocol['sensitivity'][condition]\n    ", '')
s=s.replace(",assumed_label=variant['label'],external_loss_multiplier=variant['weight']",'')
s=s.replace("if condition.startswith('B_'):\n        yy[owners>=1061]=variant['label']\n        weights=weights.copy();weights[np.repeat(owners>=1061,2)]*=variant['weight']", "if condition.startswith('B_'):yy[owners>=1061]=100")
s=s.replace("default='"+','.join(p['conditions'])+"'", "default='A,B_adult,B_child,C,D_adult,D_child'")
assert ast.dump(ast.parse(s))==ast.dump(ast.parse((OLD/'experiment.py').read_text()))
checks=[]
for f in p['folds']:
 for pilot in [False,True]:
  tr,ext,ev=e.rows_for(d,f,'B_adult',pilot);wi,w=e.example_pool(d,np.r_[tr,ext]);owners=d['window_record'][wi];mask=np.repeat(owners>=1061,2)
  assert len(np.unique(d['person_id'][ext]))==9 and len(ext)==12
  assert set(d['person_id'][tr]).isdisjoint(d['person_id'][ev])
  assert np.isnan(d['y'][ext]).all()
  sums=[d['record_weight'][ext][d['person_id'][ext]==q].sum() for q in np.unique(d['person_id'][ext])];assert np.allclose(sums,1)
  for condition,v in p['sensitivity'].items():
   tt,ee,vv=e.rows_for(d,f,condition,pilot);assert np.array_equal(tr,tt) and np.array_equal(ext,ee) and np.array_equal(ev,vv)
   ww=w.copy();ww[mask]*=v['weight'];assert np.array_equal(ww[~mask],w[~mask]);np.testing.assert_allclose(ww[mask],w[mask]*v['weight'],atol=0,rtol=0)
   yy=d['y'][owners].astype(np.float32).copy();yy[owners>=1061]=v['label'];assert np.isfinite(yy).all();assert np.array_equal(yy[owners<1061],d['y'][owners[owners<1061]].astype(np.float32))
   checks.append(dict(fold=f,pilot=pilot,condition=condition,external_people=9,stanford_examples=2*int(np.isin(d['window_record'],tr).sum()),external_weight=v['weight'],assumed_label=v['label']))
e.save(HERE/'preflight.json',dict(passed=True,checks=checks,ast_change_boundary_passed=True,model_code_identical=True,data_sha256=p['data_sha256']))
print('PREFLIGHT_PASSED',len(checks))
