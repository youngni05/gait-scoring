"""Retrieve only this round's terminated workers; keep an actual per-file receipt."""
from pathlib import Path
import importlib.util,json,tarfile,hashlib,datetime,shutil
from deploy import H,D,REMOTE,sha
ROOT=H.parents[3];REGISTRY=ROOT/'gdi-research-harness/control/gpu-watch/registry.json'
OWNER='01a18b12-4a31-7283-a35d-17f28017111a';ID='cnn-ranksim-strength-692'
# Reuse the generic, immutable packaging and SSH routines, not its three-run main.
spec=importlib.util.spec_from_file_location('existing_recovery_helpers',REGISTRY.parent/'retrieve_existing.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def current():
 r=read(REGISTRY);assert r['thread_id']==OWNER and r['automation_id']=='gdi-2'
 e=next(e for e in r['experiments'] if e['id']==ID)
 assert e['remote']==REMOTE and e['jobs']==read(H/'submission.json')['jobs']
 return r,e
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--snapshot',type=Path,required=True);a=ap.parse_args()
 reg,e=current();snap=read(a.snapshot);assert snap['thread_id']==OWNER
 state=snap['experiments'][ID];jobs=state['jobs']
 assert all(jobs[j]['state'] in helper.TERMINAL for j in e['jobs']),'Existing workers still running'
 reg,e=current()
 meta=dict(id=ID,remote=REMOTE,jobs=e['jobs'],states=jobs)
 server=json.loads(helper.command(helper.SSH+['/usr/bin/python3.8 -'],input='META='+repr(meta)+'\n'+helper.REMOTE_CODE))
 dest=D/'received'/('full-'+'-'.join(e['jobs']));dest.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(dest).free>server['archive_bytes']+1024**3
 archive=dest/'existing-results.tar.gz';inventory=dest/'server-inventory.json'
 for remote,p,digest in [(server['archive'],archive,server['archive_sha256']),(server['inventory'],inventory,server['inventory_sha256'])]:
  if not p.exists() or sha(p)!=digest:helper.command(helper.SCP+[helper.HOST+':'+remote,str(p)],timeout=600)
  assert sha(p)==digest
 body=read(inventory);expected={x['path']:x for x in body['files']};seen=set()
 with tarfile.open(archive,'r|gz') as tf:
  for m in tf:
   if not m.isfile() or m.name=='RECOVERY-INVENTORY.json':continue
   assert m.name in expected and m.name not in seen
   h=hashlib.sha256()
   with tf.extractfile(m) as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
   assert m.size==expected[m.name]['bytes'] and h.hexdigest()==expected[m.name]['sha256'];seen.add(m.name)
 assert seen==set(expected)
 succeeded=all(jobs[j]['state']=='COMPLETED' and jobs[j]['exit_code']=='0:0' for j in e['jobs'])
 full=server['counts']['main_latest']==server['counts']['main_fit_complete']==50 and state['completed_outer_folds']==10
 if succeeded:assert full and state['summary_exists'] and state['summary_complete']
 receipt=dict(passed=True,server_local_sha256_matched=True,all_archived_file_hashes_verified=True,archive=str(archive),inventory=str(inventory),archive_sha256=sha(archive),inventory_sha256=sha(inventory),counts=server['counts'],full_fit_counts_present=full,source_succeeded=succeeded,states=jobs,source_snapshot=str(a.snapshot),received_at=datetime.datetime.now().astimezone().isoformat(),performance_audit_performed=False)
 helper.save(H/'checks/full-receipt.json',receipt)
 reg,e=current();e.update(retrieval_state='received_sha256_verified',retrieval_receipt_path=str(H/'checks/full-receipt.json'),retrieval_receipt_sha256=sha(H/'checks/full-receipt.json'),performance_review_state='pending_full_cpu_audit')
 helper.save(REGISTRY,reg);print(json.dumps(receipt))
if __name__=='__main__':main()
