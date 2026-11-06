"""Replay all frozen child predictors and independently aggregate age weights."""
from pathlib import Path
import argparse, collections, hashlib, json, sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=HERE.parents[3]
BUNDLE=ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
sys.path.insert(0,str(BUNDLE))
from model import GdiCNN, channels
from infer import verify_bundle, validate_input

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--analysis-root',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(4);verify_bundle()
    d=load(args.prepared);validate_input(d)
    assert read(HERE/'decode_final_verification.json')['passed']
    assert read(HERE/'pose_retrieval_verification.json')['passed']
    qa=read(HERE/'target_final_review.json');assert qa['reviewed_all_46']
    assert set(d['record_id']).issubset(set(qa['retained_records']))
    plan=read(HERE/'age_matching_plan.json');pids=sorted(set(d['person_id']))
    counts=collections.Counter(int(plan['person_age'][p]) for p in pids)
    target=plan['development_year_counts'];represented=sum(target[str(a)] for a in counts)
    weights=np.array([target[str(int(plan['person_age'][p]))]/counts[int(plan['person_age'][p])]/represented for p in pids])
    assert abs(weights.sum()-1)<1e-12
    result=dict(passed=True,input_sha256=sha(args.prepared),people=len(pids),records=len(d['record_id']),windows=len(d['x']),age_counts=dict(counts),target_people=428,represented_target_people=represented,full_age5to12_coverage=len(counts)==8,missing_ages=[a for a in range(5,13) if a not in counts],actual_GDI_available=False,continuous_walking_validation=False,models={})
    baseline_metadata=read(BUNDLE/'models.json')
    summaries={}
    for mode in ['baseline','high90_x3','rare_sqrt_cap4']:
        folder=args.analysis_root/mode;saved=load(folder/'predictions.npz');report=read(folder/'comparison.json')
        assert report['input_sha256']==sha(args.prepared)
        np.testing.assert_array_equal(saved['person_id'],d['person_id']);np.testing.assert_array_equal(saved['record_id'],d['record_id'])
        if mode!='baseline':
            audit=read(STUDY/mode/'analysis/comparison.json');assert audit['verified_fits']==25 and len(audit['fold_replay'])==5
        errors=[]
        for f in range(5):
            if mode=='baseline':
                item=baseline_metadata['folds'][f];path=BUNDLE/item['file'];coef=item['calibration_slope_intercept'];dropout=baseline_metadata['dropout']
            else:
                received=(STUDY/mode/'received') if mode=='high90_x3' else Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-cnn-high-gdi/rare_sqrt_cap4/received')
                fd=received/'results'/('fold%d'%f);complete=read(fd/'complete.json');path=fd/'refit/latest.pt';coef=complete['calibration'];dropout=read(STUDY/mode/'bundle/protocol.json')['dropout']
                assert sha(path)==complete['checkpoint_sha256']
            ck=torch.load(path,map_location='cpu',weights_only=True);m=GdiCNN(dropout);m.load_state_dict(ck['model']);m.eval()
            parts=[]
            with torch.no_grad():
                for start in range(0,len(d['x']),128):
                    x=channels(torch.from_numpy(d['x'][start:start+128])).reshape(-1,12,124)
                    parts.append((m(x)[:,0]*ck['span']+ck['lo']).reshape(-1,2).numpy())
            raw=np.concatenate(parts).astype(np.float64)
            replay=np.stack([raw[d['window_record']==i].mean(axis=0) for i in range(len(d['record_id']))])*coef[0]+coef[1]
            error=float(np.max(np.abs(replay-saved['predictions'][f])));assert error<=1e-4;errors.append(error)
            np.testing.assert_allclose(raw*coef[0]+coef[1],saved['window_predictions'][f],rtol=0,atol=1e-4)
        scores=np.array([saved['predictions'][:,d['person_id']==p,:].mean() for p in pids])
        simple=float(scores.mean());weighted=float(scores@weights)
        assert abs(simple-report['summary']['unweighted_person_mean'])<1e-10
        assert abs(weighted-report['summary']['age_adjusted_person_mean'])<1e-10
        result['models'][mode]=dict(unweighted_mean=simple,age_weighted_mean=weighted,person_median=float(np.median(scores)),person_range=[float(scores.min()),float(scores.max())],maximum_single_fold_side_prediction=float(saved['predictions'].max()),replay_max_errors=errors,comparison_sha256=sha(folder/'comparison.json'),participants=[dict(person=p,age=plan['person_age'][p],score=float(v),weight=float(w)) for p,v,w in zip(pids,scores,weights)],movement_diagnostics=report['summary']['movement_diagnostics'])
        summaries[mode]=scores
    for mode in ['high90_x3','rare_sqrt_cap4']:
        diff=summaries[mode]-summaries['baseline'];result['models'][mode].update(unweighted_change=float(diff.mean()),age_weighted_change=float(diff@weights))
    result.update(decode_audit_sha256=sha(HERE/'decode_final_verification.json'),retrieval_audit_sha256=sha(HERE/'pose_retrieval_verification.json'),target_review_sha256=sha(HERE/'target_final_review.json'),age_plan_sha256=sha(HERE/'age_matching_plan.json'),quality_sha256=sha(HERE/'quality.json'),output_diagnostics_sha256=sha(HERE/'child_output_diagnostics.json'),primary_keeps_all_prespecified_eligible_people=True,clinical_validity_established=False)
    (HERE/'child_final_comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='models'}))
    for mode,v in result['models'].items():print(mode,v['unweighted_mean'],v['age_weighted_mean'],max(v['replay_max_errors']))

if __name__=='__main__':main()
