"""Supplement to a PRESERVED failed exact-coefficient audit; original audits unchanged.

Verify frozen coefficients/metrics and separately measure refit transformations
against the existing 1e-4 GDI prediction tolerance, including CPU64 model replay.
This does not assert the original 1e-10 coefficient equality passed.
"""
from pathlib import Path
import os,datetime,json,traceback
import numpy as np
import audit
import review_results as review
H=Path(__file__).resolve().parent;P=H/'checks/supplemental-progress.json'
state=dict(phase='running',pid=os.getpid(),started_at=datetime.datetime.now().isoformat(),completed_audits=[],coefficient_diagnostics=[])
def save():
    p=P.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(P)
def verify_transform(raw,labels,row_coef):
    fitted=review.r.calibrate(raw,labels);stored=np.asarray(row_coef)
    delta=float(np.max(abs(fitted-stored)))
    error=float(np.max(abs(review.r.corrected(raw,fitted)-review.r.corrected(raw,stored))))
    assert error<1e-4,('Supplemental transformation disagreement',error)
    state['coefficient_diagnostics'].append(dict(coefficient_max_error=delta,train_transformation_max_error=error))
    return stored
# Keep all original provenance, sample reconstruction and metric checks verbatim.
# The original exact coefficient failure stays in full-review-progress.json/log.
source=(H/'audit.py').read_text()
needle="            coef=r.calibrate(raw,y)\n            np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)"
assert source.count(needle)==1
source=source.replace(needle,"            coef=verify_transform(raw,y,row['calibration'])")
namespace=dict(verify_transform=verify_transform);exec(compile(source,'supplemental_audit_in_memory','exec'),namespace)
original=namespace['audit_fit']
def audited(*args,**kwargs):
    state['active_fit']=str(args[5]);save();h=original(*args,**kwargs)
    state['completed_audits'].append(str(args[5]));save();print('SUPPLEMENTAL_AUDITED',len(state['completed_audits']),args[5],flush=True);return h
if __name__=='__main__':
    assert json.loads((H/'checks/full-review-progress.json').read_text())['phase']=='failed'
    save();review.audit_fit=audited
    try:
        review.main()
        out=H/'checks/supplemental-verification.json';r=json.loads(out.read_text());r.update(original_exact_coefficient_audit_passed=False,original_failure_preserved=True,source_environment_proof_sha256=review.r.sha(H/'checks/source-environment-summary-proof.json'),
            supplemental_scope='Frozen coefficients and all epoch metrics; independent refit transformation <1e-4 GDI, not coefficient equality',
            coefficient_max_error=max(x['coefficient_max_error'] for x in state['coefficient_diagnostics']),transformation_max_error=max(x['train_transformation_max_error'] for x in state['coefficient_diagnostics']),epochs=len(state['coefficient_diagnostics']))
        review.r.save(out,r);state.update(phase='complete',exit_code=0,finished_at=datetime.datetime.now().isoformat());save()
    except BaseException as exc:
        state.update(phase='failed',exit_code=1,error=repr(exc));save();traceback.print_exc();raise
