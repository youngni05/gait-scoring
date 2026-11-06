"""Verify seed-only training changes using source AST fingerprints."""
import ast,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 p=read(H/'source-provenance.json');c=read(H/'training.json');plan=read(H/'protocol.json')
 assert sha(H/'protocol.json')==p['plan_sha256']
 assert c.keys()==p['old_training'].keys()
 assert all(c[k]==v for k,v in p['old_training'].items() if k not in p['allowed_training_metadata_changes'])
 assert c['seed']==plan['new_seed']==20261102
 tree=ast.parse((H/'run.py').read_text(encoding='utf-8'))
 functions={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
 # AST serialization differs across Python versions. Parse both hash-bound
 # source files in the current interpreter, retaining the same semantic check.
 assert sha(H/'source-run.py.txt')==p['source_run_sha256']
 original={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((H/'source-run.py.txt').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
 for n in p['unchanged_function_ast_sha256']:assert functions[n]==original[n],n
 for n,d in p['unchanged_files'].items():assert sha(H/n)==d,n
 return dict(passed=True,unchanged_functions=len(p['unchanged_function_ast_sha256']),new_seed=c['seed'])
if __name__=='__main__':print(json.dumps(verify()))
