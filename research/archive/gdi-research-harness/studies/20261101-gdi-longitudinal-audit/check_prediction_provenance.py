"""Check saved model identity/proof chains, without inference or outcome scoring."""
import json
from pathlib import Path
from audit import HERE, ROOT, digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    study = ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study'
    bag = study/'inner-bagging692'
    report = read(bag/'analysis/comparison.json')
    visits = read(HERE/'development-visits.json')
    people_by_fold = {f:{r['person_id'] for r in visits if r['fold']==f} for f in range(5)}
    all_people = set().union(*people_by_fold.values())
    assert len(all_people)==692 and sum(map(len,people_by_fold.values()))==692
    cache = Path(report['predictions_path']).parent
    sources, checked = {}, []
    for family in ('CNN','STT'):
        base = (study/'cnn-multiwindow695/received/results' if family=='CNN'
                else study/'stt692/multiwindow/analysis/unpacked/results')
        for outer in range(5):
            for inner in [None]+[j for j in range(5) if j!=outer]:
                tag = 'refit' if inner is None else 'inner%d'%inner
                folder = base/('fold%d'%outer)/tag
                identity_path, complete_path = folder/'identity.json', folder/'fit_complete.json'
                identity, complete = read(identity_path), read(complete_path)
                assert identity==complete['identity']
                assert identity['outer_fold']==outer and identity['inner_fold']==inner
                expected_train = all_people - people_by_fold[outer]
                if inner is not None:
                    expected_train -= people_by_fold[inner]
                assert set(identity['train_people'])==expected_train
                assert set(identity['outer_evaluation_people'])==people_by_fold[outer]
                assert set(identity['validation_people'])==(set() if inner is None else people_by_fold[inner])
                assert not set(identity['train_people']) & people_by_fold[outer]
                if inner is not None:
                    assert not set(identity['validation_people']) & people_by_fold[outer]
                robust_path = (Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/robust-aggregation692/cache')
                               /family/('fold%d_%s.json'%(outer,tag)))
                robust = read(robust_path)
                assert robust['checkpoint_sha256']==complete['checkpoint_sha256']
                for p in (identity_path,complete_path):
                    assert robust['source_files'][str(p)]==digest(p)
                    sources[str(p)]=digest(p)
                if inner is None:
                    prediction_path = robust_path.with_suffix('.npz')
                    expected_sha = robust['cache_sha256']
                else:
                    proof_path = cache/family/('fold%d_%s.json'%(outer,tag))
                    proof = read(proof_path)
                    assert proof['outer']==outer and proof['inner']==inner and proof['family']==family
                    assert proof['checkpoint_sha256']==complete['checkpoint_sha256']
                    assert proof['protocol_sha256']==digest(bag/'protocol.json')
                    prediction_path = proof_path.with_suffix('.npz')
                    expected_sha = proof['cache_sha256']
                    sources[str(proof_path)]=digest(proof_path)
                assert report['sources'][str(prediction_path)]==expected_sha==digest(prediction_path)
                sources[str(robust_path)]=digest(robust_path)
                checked.append(dict(family=family,outer=outer,inner=inner,
                                    train_people=len(expected_train),outer_people=len(people_by_fold[outer])))
    assert len(checked)==50
    result = dict(passed=True,checked_identities=50,inner_models=40,refit_models=10,
                  outer_people_excluded_from_recorded_training=True,
                  outer_people_excluded_from_inner_validation=True,
                  cache_proof_identity_and_completion_hashes_match=True,
                  scope='CNN and STT bag constituents only; saved identities/proof chains, not a new weight replay',
                  bilateral_stt_and_combination_selection_reaudited=False,
                  model_checkpoints_reloaded=False,clinical_change_metrics_computed=False,
                  checked=checked,source_sha256=sources,
                  comparison_sha256=digest(bag/'analysis/comparison.json'),
                  script_sha256=digest(Path(__file__)))
    (HERE/'prediction-provenance-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checked','source_sha256')},indent=2))


if __name__=='__main__':
    main()
