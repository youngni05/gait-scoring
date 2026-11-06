from pathlib import Path
import json,os,subprocess,sys,tempfile,zipfile
H=Path(__file__).resolve().parent;B=H/'bundle';checks=H/'checks';checks.mkdir(exist_ok=True)
results=[]
for script in ['verify_ssl.py','verify.py','verify_nested.py','verify_windows.py']:
    output=checks/script[:-3];cmd=[sys.executable,'-u','-B',str(B/script),'--output',str(output)]
    if script!='verify_windows.py':cmd+=['--device','cpu']
    with (checks/(script+'.log')).open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Failed '+script+'; see '+str(log.name))
    results.append(json.loads((output/'verification.json').read_text()));print(script+' passed',flush=True)
with tempfile.TemporaryDirectory(dir=checks,prefix='unpacked-') as temp:
    with zipfile.ZipFile(H/'gdi-stt692-ssl.zip') as z:z.extractall(temp)
    root=Path(temp)/'gdi-stt692-ssl-20261031'
    with (checks/'independent-zip.log').open('w') as log:
        p=subprocess.run([sys.executable,'-B',str(root/'run.py'),'check','--device','cpu','--output',str(checks/'zip-check')],stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0
    with (root/'pretrain.py').open('a') as f:f.write('\n# tamper check\n')
    with (checks/'corrupt-zip.log').open('w') as log:
        p=subprocess.run([sys.executable,'-B',str(root/'run.py'),'check','--device','cpu','--output',str(checks/'zip-reject')],stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode!=0 and 'Bundle changed: pretrain.py' in (checks/'corrupt-zip.log').read_text()
result=dict(passed=True,tests=results,independent_zip_real_input_check=True,tampered_ssl_code_rejected=True,real_GDI_optimizer_steps=0,GPU_executed=False)
(H/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('ALL_LOCAL_CHECKS_PASSED',flush=True)
