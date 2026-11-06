"""Report frozen two-arm OOF comparison after complete independent reviews."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
H=Path(__file__).resolve().parent
AUDIT=H.parents[1]/'20261101-gdi-longitudinal-audit'
sys.path.insert(0,str(AUDIT))
from change_metrics import summarize
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def macro(y,p,ids):return float(np.mean([abs(p[ids==i]-y[ids==i]).mean() for i in np.unique(ids)]))
def main():
    receipt=read(H/'checks/full-receipt.json');assert receipt['passed'] and len(receipt['seeds'])==2
    metadata=read(H/'seed0/visit_metadata.json');pair_specs=read(AUDIT/'prepared-pairs.json')['pairs']
    output={};all_arrays=[];sources={}
    for arm in (0,1):
        rec=next(r for r in receipt['seeds'] if r['seed']==arm)
        assert read(rec['review'])['passed'] and sha(rec['review'])==rec['review_sha256']
        path=Path(rec['extracted'])/'runs/main/oof_predictions.npz';sources[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in ('y','predictions','person_id','video_id','fold')}
        assert len(d['y'])==1061 and len(np.unique(d['person_id']))==692
        assert [r['video_id'] for r in metadata]==d['video_id'].tolist()
        assert [r['person_id'] for r in metadata]==d['person_id'].tolist()
        if all_arrays:
            for k in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(d[k],all_arrays[0][k])
        all_arrays.append(d);p=d['predictions'][:,0];index={v:i for i,v in enumerate(d['video_id'])};rows=[]
        for r in pair_specs:
            a,b=index[r['lower_age_video_ids'][0]],index[r['higher_age_video_ids'][0]]
            assert d['person_id'][a]==d['person_id'][b]==r['person_id'] and d['fold'][a]==d['fold'][b]
            rows.append(dict(person_id=r['person_id'],true_delta_lr=(d['y'][b]-d['y'][a]).tolist(),predicted_delta_lr=(p[b]-p[a]).tolist()))
        first=[]
        for person in np.unique(d['person_id']):
            ix=np.flatnonzero(d['person_id']==person)
            first.append(int(ix[0]) if len(ix)==1 else min(ix,key=lambda i:metadata[i]['age']))
        first=np.asarray(first);assert len(first)==692
        sides={}
        for s,name in enumerate(['left','right']):
            rr=[dict(r,true_delta_lr=[r['true_delta_lr'][s]]*2,predicted_delta_lr=[r['predicted_delta_lr'][s]]*2) for r in rows]
            sides[name]=summarize(rr);sides[name]['changed_side_pair_entries']//=2
        output['arm%d'%arm]=dict(change=summarize(rows),sides=sides,
                                first_available_or_lower_age_visit_mae=macro(d['y'][first],p[first],d['person_id'][first]),
                                all_visit_mae=macro(d['y'],p,d['person_id']))
    report=dict(passed=True,arms=output,people=692,change_people=229,change_pairs=581,
                paired_minus_baseline_change_mae=output['arm1']['change']['person_mean_absolute_delta_error']-output['arm0']['change']['person_mean_absolute_delta_error'],
                paired_minus_baseline_first_visit_mae=output['arm1']['first_available_or_lower_age_visit_mae']-output['arm0']['first_available_or_lower_age_visit_mae'],
                success_threshold=None,clinical_validation=False,independent_final_validation=False,
                source_sha256=sources,full_receipt_sha256=sha(H/'checks/full-receipt.json'),script_sha256=sha(Path(__file__)))
    (H/'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
