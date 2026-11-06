"""Exercise pilot result review using labelled synthetic people, never real GDI fitting."""
from pathlib import Path
import sys,tempfile,math,numpy as np,torch
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'bundle'))
import run as r
import review_pilot
from bags import predict_details,weight_summary
def main():
    rng=np.random.default_rng(45);n=10
    d=dict(y=rng.uniform(40,100,(n,2)),person_id=np.array(['synthetic%d'%i for i in range(n)]),fold=np.repeat(np.arange(5),2))
    owner=np.repeat(np.arange(n),np.tile([1,2],5));d['x']=r.WindowSet(rng.normal(size=(len(owner),2,124,4,2)).astype(np.float32),owner,n)
    tr,va,te=r.split(d,0,1);review_pilot.r.data=lambda:d
    cbase=r.load(r.ROOT/'protocol.json');checks=H/'checks';checks.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checks,prefix='pilot-review-synthetic-') as temp:
        results={}
        for mode in ['mean','attention']:
            c=dict(cbase,pooling=mode,name='record_loss_STT_'+mode);out=Path(temp)/mode
            ident=dict(bundle_sha256=r.sha(r.ROOT/'manifest.json'),seed=c['seed']+1,epochs=3,outer_fold=0,inner_fold=1,
                train_people=np.unique(d['person_id'][tr]).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),outer_evaluation_people=np.unique(d['person_id'][te]).tolist())
            model,lo,span,h=r.fit(d['x'][tr],d['y'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],c,3,out/'fit',ident,'cpu')
            raw,detail=predict_details(model,d['x'][va],lo,span,'cpu');pred=r.corrected(raw,h[-1]['calibration'])
            np.savez_compressed(out/'validation_predictions.npz',raw=raw,predictions=pred,y=d['y'][va],person_id=d['person_id'][va],row_index=np.flatnonzero(va))
            np.savez_compressed(out/'window_weights.npz',**detail,row_index=np.flatnonzero(va));r.save(out/'weight_summary.json',weight_summary(d['x'][va],detail))
            r.save(out/'pilot_summary.json',dict(pilot_only=True,variant=mode,outer_evaluation_used=False,epochs=3,optimizer_steps=3*math.ceil(int(tr.sum())/c['batch_records']),checkpoint_sha256=r.sha(out/'fit/latest.pt'),validation_person_mae=r.metric(d['y'][va],pred,d['person_id'][va])))
            sys.argv=['review_pilot.py','--variant',mode,'--device','cpu','--input',str(out),'--output',str(out/'review')]
            review_pilot.main();v=r.load(out/'review/verification.json');assert v['passed'];results[mode]=v
            # A wrong saved prediction must not be accepted as a successful replay.
            path=out/'validation_predictions.npz'
            with np.load(path) as z:altered={k:z[k] for k in z.files}
            altered['raw']=altered['raw']+1.;np.savez_compressed(path,**altered)
            try:review_pilot.main()
            except AssertionError:pass
            else:raise AssertionError('Tampered pilot predictions accepted')
        r.save(checks/'pilot-review/verification.json',dict(passed=True,synthetic_only=True,real_GDI_optimizer_steps=0,both_variants_reviewed=True,tampered_pilot_predictions_rejected=True,variants=results))
if __name__=='__main__':main()
