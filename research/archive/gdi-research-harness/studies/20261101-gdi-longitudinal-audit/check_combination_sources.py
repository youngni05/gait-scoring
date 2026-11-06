"""Audit bilateral model identities and frozen combination selection evidence."""
import hashlib
import json
import zipfile
from pathlib import Path
from audit import HERE,ROOT,array,digest

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stream_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    study=ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study'
    visits=read(HERE/'development-visits.json')
    groups={f:{r['person_id'] for r in visits if r['fold']==f} for f in range(5)}
    all_people=set().union(*groups.values());assert len(all_people)==692
    model_count=0;sources={};byte_count=0
    base=study/'stt692/bilateral/analysis/unpacked/results'
    for outer in range(5):
        for inner in [None]+[i for i in range(5) if i!=outer]:
            folder=base/('fold%d'%outer)/('refit' if inner is None else 'inner%d'%inner)
            identity,done=read(folder/'identity.json'),read(folder/'fit_complete.json')
            assert identity==done['identity']
            assert identity['outer_fold']==outer and identity['inner_fold']==inner
            train=all_people-groups[outer]-(groups[inner] if inner is not None else set())
            assert set(identity['train_people'])==train
            assert set(identity['validation_people'])==(groups[inner] if inner is not None else set())
            assert set(identity['outer_evaluation_people'])==groups[outer]
            assert not train & groups[outer]
            for name in ['identity.json','fit_complete.json']:
                p=folder/name;sources[str(p)]=digest(p)
            model_count+=1
    selection_reports={}
    for name in ['cnn-stt-ensemble692','mean-difference692']:
        directory=study/name/'analysis';selection=read(directory/'selection.json');verification=read(directory/'verification.json')
        assert verification['passed'] and not selection['outer_predictions_opened']
        assert selection['protocol_sha256']==digest(study/name/'protocol.json')
        if name=='mean-difference692':assert verification['selection_sha256']==digest(directory/'selection.json')
        for key,expected in selection['source_hashes'].items():
            path=Path(key) if Path(key).is_absolute() else study/key
            if str(path) not in sources:
                actual=stream_sha(path);sources[str(path)]=actual;byte_count+=path.stat().st_size
            assert sources[str(path)]==expected
        for f in selection['folds']:
            outer=f['outer'];p=directory/('inner_outer%d.npz'%outer)
            with zipfile.ZipFile(p) as z:
                ids,folds=array(z,'person_id'),array(z,'fold')
            assert len(ids)==len(folds)
            assert set(ids)==all_people-groups[outer] and outer not in folds
            assert not set(ids)&groups[outer]
            expected=f['inner_sha256'] if name=='mean-difference692' else f['inner_file_sha256']
            assert digest(p)==expected
            sources[str(p)]=expected
        selection_reports[name]=dict(source_references=len(selection['source_hashes']),outer_fold_selection_caches=5,
                                     outer_people_absent=True,protocol_sha256=selection['protocol_sha256'])
    result=dict(passed=True,bilateral_model_identities=model_count,
                inner_models=20,refit_models=5,selection_reports=selection_reports,
                unique_source_files_checked=len(sources),additional_source_bytes_streamed=byte_count,
                checkpoints_hashed_when_referenced=True,checkpoints_reloaded=False,
                score_metrics_recomputed=False,clinical_or_causal_validation=False,
                scope='Saved identities, current source hashes and selection-cache person boundaries; no training/inference/poison-test replay',
                sources=sources,script_sha256=digest(Path(__file__)))
    (HERE/'combination-source-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))

if __name__=='__main__':main()
