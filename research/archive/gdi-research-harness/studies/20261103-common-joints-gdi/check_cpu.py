import os,subprocess,sys,json
from pathlib import Path
h=Path(__file__).resolve().parent;b=h/'bundle';big=Path('D:/gait-research-data/D_P/common-joints-gdi-20261103')
result=[]
for variant in ['full124','common124','full90','common90']:
 env=dict(os.environ,GDI_VARIANT=variant,GDI_DATA_PATH=str(big/'data.npz'),PYTHONDONTWRITEBYTECODE='1')
 out=big/'checks/cpu'/variant;out.mkdir(parents=True,exist_ok=True)
 with (out/'log.txt').open('w') as f:p=subprocess.run([sys.executable,str(b/'check_one.py'),'--device','cpu','--output',str(out),'--synthetic-full'],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(variant,'exit',p.returncode,flush=True)
 if p.returncode:print((out/'log.txt').read_text()[-3000:]);raise SystemExit(p.returncode)
 result.append(json.loads((out/'check.json').read_text()))
(h/'cpu-verification.json').write_text(json.dumps(dict(passed=True,conditions=result),indent=2))
