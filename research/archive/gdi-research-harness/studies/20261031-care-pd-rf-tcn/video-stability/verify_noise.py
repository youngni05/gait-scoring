"""Strict replay of perturbations using the inference platform's NumPy."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
protocol=json.loads((HERE/'protocol.json').read_text())
complete=json.loads((HERE/'main/complete.json').read_text())
rows=[]
for row in complete['outputs']:
    p=HERE/'main'/(row['key']+'.npz')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    c=next(c for c in protocol['conditions'] if row['key'].endswith('_'+c['name']))
    expected=np.clip(np.random.RandomState(c['seed']).normal(size=(150,3)),-3,3)*c['sigma']
    z=np.load(p);actual=z['noise'];assert np.array_equal(actual,expected)
    rows.append(dict(key=row['key'],file_sha256=row['sha256'],expected_noise_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),exact_match=True))
result=dict(passed=True,numpy=np.__version__,python=sys.version,outputs=len(rows),maximum_difference=0.,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/'noise-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
