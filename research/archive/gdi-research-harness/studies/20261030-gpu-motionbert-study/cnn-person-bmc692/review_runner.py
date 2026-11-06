"""Expose progress around unchanged full audits, so later hourly runs can resume monitoring."""
from pathlib import Path
import os,json,sys,datetime,traceback
import review_results as review
H=Path(__file__).resolve().parent;P=H/'checks/full-review-progress.json'
original=review.audit_fit
state=dict(phase='running',pid=os.getpid(),started_at=datetime.datetime.now().isoformat(),completed_audits=[])
def save():
    tmp=P.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(P)
def audit(*args,**kwargs):
    state['active_fit']=str(args[6]);save();result=original(*args,**kwargs)
    state['completed_audits'].append(str(args[6]));save();print('AUDITED',len(state['completed_audits']),args[6],flush=True);return result
if __name__=='__main__':
    save();review.audit_fit=audit
    try:
        review.main();state.update(phase='complete',exit_code=0,finished_at=datetime.datetime.now().isoformat());save()
    except BaseException as exc:
        state.update(phase='failed',exit_code=1,error=repr(exc));save();traceback.print_exc();raise
