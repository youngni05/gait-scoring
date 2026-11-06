"""Bind the completed original summaries to their mandatory all-fit audits."""
import ast
assert receipt['source_succeeded'] and all(q=={'state':'COMPLETED','exit_code':'0:0'} for q in receipt['states'].values())
marker=read(U/'runs/summary_complete.json');pair=read(U/'runs/pair_summary.json')
assert sha(U/'runs/pair_summary.json')==marker['summary_sha256'] and pair['fits']==50 and pair['outer_folds']==10
assert sha(U/'runs/strength_selection.json')==pair['strength_selection_sha256']
workers=[read(U/'control'/('worker-%d.json'%i)) for i in range(3)]
assert {w['job_id'] for w in workers}=={'33677','33678','33679'}
assert len({w['uuid'] for w in workers})==3 and all(w['phase']=='assigned_folds_complete' and w['exit_code']==0 for w in workers)
tasks=[tuple(t) for w in workers for t in w['assignments']]
assert len(tasks)==len(set(tasks))==10 and set(tasks)=={(v,f) for v in ['rank01','rank03'] for f in range(5)}
def function_ast(path,name):
 return next(ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name)
assert function_ast(U/'run.py','summarize')==function_ast(S/'cnn-ranksim692/run.py','summarize')
assert sha(U/'audit.py')==sha(S/'cnn-ranksim692/audit.py')
fits=0;epochs=0
for variant in ['rank01','rank03']:
 summary=read(U/'runs/main'/variant/'summary.json')
 assert summary==pair['variants'][variant] and summary['fits']==25
 assert sha(U/'runs/main'/variant/'oof_predictions.npz')==summary['oof_sha256']
 assert abs(summary['metrics'][0]['mae']-models[variant]['mae'])<1e-10
 curves=list((U/'runs/main'/variant).glob('fold*/*/curve.json'));assert len(curves)==25
 fits+=len(curves);epochs+=sum(len(read(p)) for p in curves)
environments=[read(U/'checks'/('gpu-rank%d'%i)/'verification.json')['environment'] for i in range(3)]
assert all(e['numpy']=='1.24.4' and e['torch']=='2.4.0+cu121' for e in environments)
save(H/'checks/source-environment-summary-proof.json',dict(passed=True,summary_fits=fits,epochs=epochs,
 manifest_sha256=sha(H/'manifest.json'),source_marker_sha256=sha(U/'runs/summary_complete.json'),
 worker_jobs=[w['job_id'] for w in workers],environments=environments,source_numpy_coefficient_tolerance=1e-10,
 proof_basis='Successful original manifest-bound worker summaries each invoke unchanged original audit_fit on all25fits before writing results; source code, summaries, marker and termination verified',
 fresh_server_reexecution=False,source_results_modified=False))
