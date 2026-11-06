"""Synthetic nested evaluation, refit, resume and held-out isolation checks."""
from pathlib import Path
import argparse,tempfile,json
import numpy as np
import torch
import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);run.setup(19,a.device)
    c=run.load(run.ROOT/'protocol.json');c=dict(c,max_epochs=3)
    rng=np.random.default_rng(19);n=20
    d=dict(x=rng.normal(size=(n,2,124,4,2)).astype(np.float32),y=rng.uniform(40,100,(n,2)),
        person_id=np.repeat(np.array(['p%d'%i for i in range(10)]),2),video_id=np.array(['v%d'%i for i in range(n)]),
        fold=np.repeat(np.arange(5),4),cohort=np.zeros(n,dtype=int),original_row_index=np.arange(n))
    # Weighted choice differs from unweighted choice; ties must choose earliest.
    hs=[[dict(validation_people=w,validation_person_mae=v) for v in vals] for w,vals in [(1,[0.,10.]),(1,[0.,10.]),(1,[0.,10.]),(10,[10.,0.])]]
    assert run.select_epoch(hs)[0]==2
    assert run.select_epoch([[dict(validation_people=1,validation_person_mae=1.)]*3 for _ in range(4)])[0]==1
    with tempfile.TemporaryDirectory(dir=a.output,prefix='nested-') as temp:
        root=Path(temp);full=root/'full'
        for f in range(5):run.train_fold(d,c,f,full,a.device)
        run.summarize(d,c,full)
        summary=run.load(full/'summary.json');assert summary['fits']==25 and summary['metrics']['people']==10
        old=[run.sha(q) for q in sorted((full/'fold0').glob('*/latest.pt'))]
        run.train_fold(d,c,0,full,a.device)
        assert old==[run.sha(q) for q in sorted((full/'fold0').glob('*/latest.pt'))]
        altered={k:v.copy() for k,v in d.items()};altered['x'][d['fold']==0]+=100;altered['y'][d['fold']==0]+=1000
        run.train_fold(altered,c,0,root/'altered',a.device)
        for fit in ['inner1','inner2','inner3','inner4','refit']:
            left=torch.load(full/'fold0'/fit/'latest.pt',map_location='cpu',weights_only=True)
            right=torch.load(root/'altered/fold0'/fit/'latest.pt',map_location='cpu',weights_only=True)
            assert left['identity']==right['identity'] and left['lo']==right['lo'] and left['span']==right['span']
            for k,v in left['model'].items():assert torch.equal(v,right['model'][k])
            assert [h['calibration'] for h in left['history']]==[h['calibration'] for h in right['history']]
        assert run.load(full/'fold0/selection.json')==run.load(root/'altered/fold0/selection.json')
        target=full/'fold0/refit/raw_predictions/epoch0001.npz';target.write_bytes(target.read_bytes()+b'corrupt')
        try:run.summarize(d,c,full)
        except ValueError as e:assert 'Raw prediction hash mismatch' in str(e)
        else:raise AssertionError('Tampered diagnostic accepted')
    result=dict(passed=True,synthetic_only=True,real_GDI_optimizer_steps=0,device=a.device,nested_25_fits_completed=True,
        weighted_selection_and_earliest_tie_verified=True,outer_input_and_label_changes_do_not_change_models_or_selection=True,
        completed_fit_resume_does_not_update_weights=True,all_epoch_raw_predictions_and_calibrations_recomputed=True,corrupt_diagnostics_rejected=True)
    run.save(a.output/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
