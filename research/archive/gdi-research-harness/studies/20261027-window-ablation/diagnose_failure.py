"""Test whether clipping explains chained-normalization mismatch; no scores."""
import importlib.util
import numpy as np
from prepare import HERE,H,ROOT,SHORT,DATA,context,load,save,require,sha
from short_window_contract import prepare_candidate,source_functions
from native124_cache import read_poses

def main():
    require((HERE/'failure_diagnosis_plan.md').is_file(),'Missing diagnostic plan')
    videos,_,_,_=context()
    index=next(i for i in range(243) if not (HERE/f'records/row_{i:04d}.npz').exists())
    tasks=load(H/'control/desktop-sessions/20261026T100042Z-9d41cd/native124_tasks.json')['tasks']
    members=load(SHORT/'zip_members.json')[tasks[index]['raw_folder']]
    spec=importlib.util.spec_from_file_location('archive_reader',ROOT/'tools/stanford-sample-check/inspect_archive.py')
    arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)
    poses,frames,_=read_poses(arc.ARCHIVE,members)
    candidate=prepare_candidate(poses,frames,243)
    start=candidate['start_index'];normalize,convert,mapping=source_functions()
    raw=convert(poses[start:start+243].astype(np.float32)[:,mapping],'coco','h36m')
    valid=raw[raw[...,2]!=0][:,:2]
    low,high=valid.min(axis=0),valid.max(axis=0);scale=float((high-low).max())
    unclipped=raw.copy();unclipped[...,:2]=(raw[...,:2]-(low+high)/2)/scale*2
    with np.load(DATA/'windows'/(str(videos[index])+'.npz'),allow_pickle=False) as z:
        original=z['input'].copy()
    with np.load(SHORT/f'records/row_{index:04d}.npz',allow_pickle=False) as z:
        correct=z['input124'].copy()
    require(np.array_equal(np.clip(unclipped,-1,1).astype(np.float32),original),'Full normalization not reproduced')
    chained=normalize(original[:124]);no_prior_clip=normalize(unclipped[:124])
    delta=np.abs(chained-correct);bad=~np.isclose(chained,correct,rtol=1e-5,atol=2e-6)
    zero=correct[...,2]==0
    result={'verified':True,'failed_row_index':index,'temporary_completed_records':len(list((HERE/'records').glob('*.npz'))),
            'chained_max_abs_difference':float(delta.max()),'mismatched_scalar_values':int(bad.sum()),
            'mismatches_at_zero_confidence_points':int(bad[zero].sum()),
            'mismatches_at_positive_confidence_points':int(bad[~zero].sum()),
            'mismatches_by_channel':bad.sum(axis=(0,1)).tolist(),
            'unclipped_then_local_max_abs_difference':float(np.max(np.abs(no_prior_clip-correct))),
            'unclipped_then_local_matches':bool(np.allclose(no_prior_clip,correct,rtol=1e-5,atol=2e-6)),
            'confidence_values_identical':bool(np.array_equal(chained[...,2],correct[...,2])),
            'GDI_labels_used':False,'model_used':False,'raw_recovery_required':True,
            'scope':'one failed existing-development record; diagnostic coordinate transformation only'}
    require(result['chained_max_abs_difference']>2e-6 and result['unclipped_then_local_matches'],'Clipping hypothesis not supported')
    save(HERE/'failed_preparation.json',{'status':'failed_before_score_training','reason':'chained normalized input not equivalent to raw124',
         'completed_temporary_records':result['temporary_completed_records'],'coordinate_max_abs_difference':result['chained_max_abs_difference'],
         'score_training_started':False,'temporary_local_features_not_used':True,'failure_criteria_relaxed':False})
    save(HERE/'normalization_failure_diagnosis.json',result)
    print(__import__('json').dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
