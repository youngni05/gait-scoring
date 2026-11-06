"""Preserve the frozen bundle; correct only the verifier's accumulation precision."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
p=HERE/'high90_x3/bundle'
a=(p/'audit.py').read_text();assert a.count('w.mean()')==1
(p/'audit64.py').write_text(a.replace('w.mean()', 'w.mean(dtype=np.float64)'),encoding='utf-8')
w=(p/'worker.py').read_text()
old="call('audit.py',['weights','--output','checks'],300)"
assert w.count(old)==1
(p/'worker_v2.py').write_text(w.replace(old,"call('audit64.py',['weights','--output','checks'],300)"),encoding='utf-8')
record={'failed_job':'33174','failure':'float32 reduction absolute error2.38418579e-7 exceeds2e-7',
        'change':'float64 mean accumulation in weight verifier only',
        'unchanged':['training tensors','loss function','tolerance2e-7','frozen manifest','training protocol','rare_sqrt_cap4 running job33175'],
        'sha256':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['audit64.py','worker_v2.py']}}
(HERE/'high90_x3/audit_repair.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))
