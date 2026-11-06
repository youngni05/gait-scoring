"""Read-only original-environment audit; executed by analyze_completed.py."""
remote_code = r'''
import os,sys,json,platform
from pathlib import Path
root=Path('/abr/RESEARCH_USER/gdi-synth-conv692-20261101')
os.chdir(root);sys.path.insert(0,str(root))
import run as r
from audit import audit_fit
import numpy as np
import torch
r.setup(7,'cpu');r.verify_manifest();d=r.data();fits=0;epochs=0
variant=sys.argv[2];f=int(sys.argv[3]);assert variant in ['scratch','synth'] and f in range(5)
for variant in [variant]:
 c=r.config_for(variant,'main')
 for f in [f]:
  folder=root/'runs/main'/variant/('fold%d'%f)
  histories=[]
  for j in range(5):
   if j==f:continue
   h=audit_fit(r,d,c,f,j,folder/('inner%d'%j),c['max_epochs'],'inner_epoch_selection')
   histories.append(h);fits+=1;epochs+=len(h)
  epoch,means=r.select_epoch(histories);sel=r.load(folder/'selection.json');done=r.load(folder/'complete.json')
  assert epoch==sel['selected_epoch']==done['selected_epoch']
  np.testing.assert_allclose(means,sel['inner_mean_person_mae_by_epoch'],rtol=0,atol=1e-10)
  h=audit_fit(r,d,c,f,None,folder/'refit',epoch,'outer_refit');fits+=1;epochs+=len(h)
  assert r.sha(folder/'predictions.npz')==done['prediction_sha256']
  assert r.sha(folder/'refit/latest.pt')==done['checkpoint_sha256']
  print('ORIGINAL_ENV_FIT',variant,f,fits,flush=True)
assert fits==5
print(json.dumps(dict(passed=True,variant=variant,fold=f,fits=fits,epochs=epochs,python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),manifest_sha256=r.sha(root/'manifest.json'),coefficient_tolerance=1e-10,original_files_changed=False,new_training=False)))
'''
proof_path=H/'checks/source-environment-audit-proof.json'
if proof_path.exists():
 source_proof=read(proof_path)
 assert source_proof['passed'] and source_proof['fits']==50 and source_proof['manifest_sha256']==sha(H/'manifest.json')
else:
 auth=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
 proofs=[];source_digest=hashlib.sha256(remote_code.encode()).hexdigest()
 for variant in ['scratch','synth']:
  for fold in range(5):
   path=O/('source-audit-%s-fold%d.json'%(variant,fold))
   if path.exists():
    proof=read(path)
   else:
    state['source_active_fold']=[variant,fold];progress()
    command=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+auth+['RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 - gdi-synth-readonly-audit %s %d'%(variant,fold)]
    result=subprocess.run(command,input=remote_code,text=True,encoding='utf-8',capture_output=True,timeout=600)
    execution=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,read_only=True,script_sha256=source_digest)
    save(O/('source-audit-%s-fold%d-execution.json'%(variant,fold)),execution)
    assert result.returncode==0,result.stderr
    proof=json.loads(result.stdout.strip().splitlines()[-1]);proof['script_sha256']=source_digest
    save(path,proof)
   assert proof['passed'] and proof['fits']==5 and proof['manifest_sha256']==sha(H/'manifest.json')
   assert proof['script_sha256']==source_digest and proof['variant']==variant and proof['fold']==fold
   proofs.append(proof);print('SOURCE_AUDIT_FOLD',variant,fold,flush=True)
 source_proof=dict(passed=True,fits=sum(p['fits'] for p in proofs),epochs=sum(p['epochs'] for p in proofs),fold_proofs=proofs,manifest_sha256=sha(H/'manifest.json'),coefficient_tolerance=1e-10,original_files_changed=False,new_training=False)
 assert source_proof['fits']==50
 save(proof_path,source_proof)
