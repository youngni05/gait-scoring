"""Independent CPU seed replay in the actual training library version."""
import json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
program='''import sys,json
from pathlib import Path
root=Path('/abr/RESEARCH_USER/normal-label-sensitivity-20261105');sys.path.insert(0,str(root));import experiment as e
p=e.read(root/'protocol.json');proof=[]
for family in ['CNN','STT']:
 for f in range(5):
  seed=p['seed']+100*f+50;e.setup(seed);model=e.newmodel(family);keys=e.backbone_keys(model,family,p);s=model.state_dict()
  proof.append(dict(family=family,fold=f,seed=seed,head_sha256=e.state_digest(s,[k for k in s if k not in keys]),backbone_sha256=e.state_digest(s,keys)))
print(json.dumps(dict(manifest_sha256=e.sha(root/'manifest.json'),torch=str(e.torch.__version__),device='CPU',rows=proof)))
'''
auth=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i',str(Path.home()/'.ssh/id_ed25519_gdi_abrm02.pub')]
r=subprocess.run([str(Path(os.environ['WINDIR'])/'System32/OpenSSH/ssh.exe'),'-p','10000',*auth,'RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8','-'],input=program,text=True,capture_output=True,check=True)
proof=json.loads(r.stdout);(HERE/'server-initialization-replay.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(torch=proof['torch'],replayed=len(proof['rows'])),indent=2))
