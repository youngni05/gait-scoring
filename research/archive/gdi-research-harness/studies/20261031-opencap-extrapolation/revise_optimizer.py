from study import *
assert not list(HERE.glob('*/completion.json'))
protocol=json.loads((HERE/'protocol.json').read_text());save('protocol-initial.json',protocol)
save('optimization-note.json',{'failure':'Gamma Newton-Cholesky line search failed at iteration519 with tol1e-8; ConvergenceWarning interrupted initial run before any cohort completed','change':'same Gamma log-link objective and alpha grid, L-BFGS max_iter3000 tol1e-7; re-run all fits including partial outputs','test_performance_read':False,'changed_data_or_splits_or_selection_metric':False})
protocol['fit_code_sha256']=sha(HERE/'study.py');protocol['optimizer_revision']='Gamma L-BFGS, same model objective; see optimization-note.json';save('protocol.json',protocol)
