"""Compare the video adapter with the original frozen CARE preprocessing."""
import sys,json,ast,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));from analyze_video import feature
import numpy as np
from study import stats,MAJOR

source=HERE.parent/'prepare.py';tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['rotations','sequence','features']]
namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
rng=np.random.RandomState(20261031);errors=[]
for amplitude in [0.,.01,.5,2.]:
    pose=rng.normal(size=(150,24,3))*amplitude
    seq=namespace['sequence'](pose.reshape(150,72),30)
    cols=[6*(j-1)+k for j in MAJOR for k in range(6)]
    old=stats(seq[:,cols])[np.r_[60:120,240:308]]
    new=feature(namespace['rotations'](pose[:,1:]))
    errors.append(float(abs(old-new).max()))
    errors.append(float(abs(namespace['features'](seq)-stats(seq)).max()))
assert max(errors)<=1e-6
result=dict(passed=True,cases=4,adapters=[128,698],frames=150,fps=30,max_feature_error=max(errors),case_errors=errors,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((HERE/'analyze_video.py').read_bytes()).hexdigest(),scope='Synthetic adapter-equivalence checks only; no clinical or video reconstruction validation.')
(HERE/'preprocessing-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
