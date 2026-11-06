from pathlib import Path
import hashlib,json,sys,runpy
R=Path('C:/Users/jun/Desktop/D_P');S=R/'gdi-research-harness/control/manual-sessions/20261029-gpu-finetune-preparation';B=S/'bundle'
files=[p for p in B.rglob('*') if p.is_file() and p.name!='manifest.json']
(B/'manifest.json').write_text(json.dumps({p.relative_to(B).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
for p in [R/'outputs/motionbert-smoke-20261015/deps',R/'outputs/gdi-ridge-experiment-20261016/deps',B]:sys.path.insert(0,str(p))
for action in ['check','smoke']:
 sys.argv=[str(B/'run.py'),action,'--device','cpu','--output',str(S/('cpu_v2_'+action))];runpy.run_path(str(B/'run.py'),run_name='__main__')
sys.argv=[str(B/'verify_synthetic.py'),'--output',str(S/'cpu_integration_v2')];runpy.run_path(str(B/'verify_synthetic.py'),run_name='__main__')
