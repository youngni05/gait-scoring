"""Prepare the frozen seed-only replication without changing any previous bundle."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent;OLD=H.parent/'cnn-ranksim692';PREV=H.parent/'cnn-ranksim-strength692'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (H/'submission.json').exists(),'Submitted bundle is immutable'
plan=read(H/'protocol.json');assert plan['new_seed']==20261102 and sha(OLD/'manifest.json')==plan['source_manifest_sha256']
for n,d in read(OLD/'manifest.json').items():assert sha(OLD/n)==d,n
names=['run.py','model.py','windows.py','balanced.py','audit.py','review.py','review_results.py','verify.py','rank_aux.py','ranking.py','ranksim.py','RANKSIM_LICENSE','reference_metrics.json','input_contract.json']
for n in names:shutil.copyfile(OLD/n,H/n)
shutil.copyfile(OLD/'run.py',H/'source-run.py.txt')
training=read(OLD/'protocol.json');training.update(seed=plan['new_seed'],name='cnn_person_mse_ranksim_seed20261102',note='Seed-only replication; frozen separate protocol.json defines evaluation and gates')
save(H/'training.json',training)
s=(H/'run.py').read_text(encoding='utf-8');assert s.count("load(ROOT / 'protocol.json')")==1
(H/'run.py').write_text(s.replace("load(ROOT / 'protocol.json')","load(ROOT / 'training.json')"),encoding='utf-8')
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
old,new=functions(OLD/'run.py'),functions(H/'run.py')
assert old.keys()==new.keys()
assert all(old[k]==new[k] for k in old if k!='config')
save(H/'source-provenance.json',dict(source_manifest_sha256=sha(OLD/'manifest.json'),source_run_sha256=sha(OLD/'run.py'),unchanged_function_ast_sha256={k:hashlib.sha256(old[k].encode()).hexdigest() for k in old if k!='config'},unchanged_files={n:sha(OLD/n) for n in names if n not in ['run.py','verify.py']},old_training=read(OLD/'protocol.json'),allowed_training_metadata_changes=['seed','name','note'],plan_sha256=sha(H/'protocol.json'),only_training_factor='RNG base seed20261028 to20261102'))
# Keep the proven synchronization/locks/local gate, with two original loss arms.
s=(PREV/'worker.py').read_text(encoding='utf-8').replace("['rank01','rank03']","['mse','ranksim']")
s=s.replace("            execute(['select_strength.py'])\n",'').replace(",strength_selection_sha256=r.sha(root/'runs/strength_selection.json')",'')
s=s.replace("phase='checking',started_at=datetime.datetime.now().isoformat()","phase='checking',started_at=datetime.datetime.now().isoformat(),started_epoch=time.time()")
s=s.replace('RANK_STRENGTH_WORKER_FINISHED','RANK_SEED_WORKER_FINISHED')
(H/'worker.py').write_text(s,encoding='utf-8')
# Deployment and retrieval are isolated to this experiment; reuse packaging only.
for n in ['deploy.py','receive_full.py','receive_pilot.py']:
 s=(PREV/n).read_text(encoding='utf-8').replace('cnn-ranksim-strength692','cnn-ranksim-seed692').replace('cnn-ranksim-strength-692','cnn-ranksim-seed-692').replace('gdi-cnn-ranksim-strength692-20261101','gdi-cnn-ranksim-seed692-20261102')
 # Replace the date after the shorter path substitution above.
 s=s.replace('gdi-cnn-ranksim-seed692-20261101','gdi-cnn-ranksim-seed692-20261102').replace('gdi-rank-strength-','gdi-rank-seed-')
 if n=='deploy.py':s=s.replace("synthetic_nested_fits']==75","synthetic_nested_fits']==50").replace("root/'protocol.json'","root/'training.json'")
 if n=='receive_pilot.py':
  s=s.replace("['rank01','rank03']","['mse','ranksim']")
  s=s.replace("workers={p.name:json.loads(p.read_text()) for p in (h/'control').glob('worker-*.json')}","workers={p.name:json.loads(p.read_text()) for p in (h/'control').glob('worker-*.json')},server_epoch=__import__('time').time()")
  needle='    # Approve only the exact manifest and pilot evidence replayed locally.'
  s=s.replace(needle,"    from runtime_gate import evaluate\n    forecast=evaluate(extracted,remote)\n    (H/'checks/runtime-estimate.json').write_text(json.dumps(forecast,indent=2)+'\\n')\n    assert forecast['passed'],'Remaining allocation cannot safely fit projected main tasks; preserve waiting workers and report'\n"+needle)
 s=s.replace('gdi-cnn-ranksim-seed692-20261102','gdi-cnn-ranksim-seed692-v2-20261102').replace('gdi-rank-seed-','gdi-rank-seed-v2-')
 (H/n).write_text(s,encoding='utf-8')
s=(H/'verify.py').read_text(encoding='utf-8').replace("env=r.setup(42,a.device);c=r.config();d=r.data()","env=r.setup(42,a.device);c=r.config();d=r.data()\n    from source_contract import verify\n    verify()")
(H/'verify.py').write_text(s,encoding='utf-8')
for p in H.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
print(json.dumps(dict(prepared=True,unchanged_run_functions=len(old)-1,training_seed=training['seed'],plan_sha256=sha(H/'protocol.json'))))
