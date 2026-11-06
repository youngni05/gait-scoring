"""Package verified code to D, transfer, verify and submit ONCE."""
from pathlib import Path
import hashlib,json,subprocess,zipfile,datetime
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
REMOTE='/abr/RESEARCH_USER/gdi-cnn-ordinal692-20261031'
D=Path('D:/gait-research-data/D_P')/H.relative_to(ROOT);D.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
proof=json.loads((H/'checks/verification.json').read_text());assert proof['passed'] and proof['manifest_sha256']==sha(H/'manifest.json')
assert not (H/'submission.json').exists(),'Already submitted or attempted; inspect before retry'
files=json.loads((H/'manifest.json').read_text())
for n,dig in files.items():assert sha(H/n)==dig
source=H.parent/'cnn-record-loss692/bundle/data.npz';assert sha(source)==json.loads((H/'protocol.json').read_text())['data_sha256']
archive=D/'gdi-cnn-ordinal692-20261031.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n in [*files,'manifest.json','README.md','checks/verification.json','checks/baselines.json']:z.write(H/n,'gdi-cnn-ordinal692-20261031/'+n)
    z.write(source,'gdi-cnn-ordinal692-20261031/data.npz')
opts=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
def ssh(code):return subprocess.run(['ssh',*opts,'-p','10000','RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 -'],input=code,text=True,capture_output=True,check=True).stdout
assert json.loads(ssh("from pathlib import Path\nimport json\nprint(json.dumps(not Path('"+REMOTE+"').exists()))"))
subprocess.run(['scp',*opts,'-P','10000',str(archive),'RESEARCH_USER@PRIVATE_HOST:/abr/RESEARCH_USER/gdi-cnn-ordinal692-20261031.zip'],check=True)
code='''from pathlib import Path
import hashlib,json,zipfile,subprocess
p=Path('/abr/RESEARCH_USER/gdi-cnn-ordinal692-20261031.zip')
assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
with zipfile.ZipFile(p) as z:z.extractall('/abr/RESEARCH_USER')
root=Path(%r)
for name,digest in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256((root/'data.npz').read_bytes()).hexdigest()==%r
(root/'logs').mkdir()
q=subprocess.run(['sbatch','--parsable','submit.slurm'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
print(json.dumps(dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)))
'''%(sha(archive),REMOTE,sha(source))
result=json.loads(ssh(code));record=dict(remote=REMOTE,archive=str(archive),archive_sha256=sha(archive),manifest_sha256=sha(H/'manifest.json'),submitted_at=datetime.datetime.now().isoformat(),submission=result,gpu_count=2,workers=2)
if result['returncode']==0:record['job_id']=result['stdout'].strip().split(';')[0];assert record['job_id'].isdigit()
save(H/'submission.json',record);print(json.dumps(record),flush=True)
assert result['returncode']==0,result
