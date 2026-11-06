"""Independent person-weighted metrics and scalar reconstruction of the fixed average."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;OLD=H.parent/'20261030-gpu-motionbert-study'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    report=read(H/'comparison.json');assert report['passed'] and not report['seed_or_weight_selection_on_outer'] and not report['independent_final_evaluation']
    assert report['protocol_sha256']==sha(H/'protocol.json') and report['full_receipt_sha256']==sha(H/'checks/full-receipt.json')
    receipt=read(H/'checks/full-receipt.json');assert receipt['passed'] and [e['seed'] for e in receipt['seeds']]==[20261101,20261102]
    for e in receipt['seeds']:assert e['passed'] and sha(e['review'])==e['review_sha256'] and read(e['review'])['passed']
    assert sha(report['predictions_path'])==report['predictions_sha256'];z=arrays(report['predictions_path']);y=z['y'];ids=z['person_id'];fold=z['fold'];people=np.unique(ids)
    assert y.shape==(1061,2) and len(people)==692
    weights=np.zeros((len(y),2))
    for person in people:
        m=ids==person;assert len(np.unique(fold[m]))==1;weights[m]=1/(len(people)*m.sum()*2)
    assert abs(weights.sum()-1)<1e-12
    metric_difference=[]
    for name,values in report['metrics'].items():
        p=z[name];mae=float((np.abs(p-y)*weights).sum());bias=float(((p-y)*weights).sum());metric_difference.extend([abs(mae-values['mae']),abs(bias-values['bias'])])
        for f in range(5):
            m=fold==f;metric_difference.append(abs(float((np.abs(p[m]-y[m])*weights[m]).sum()/weights[m].sum())-values['fold_mae'][f]))
        high=np.zeros(len(y),bool)
        for person in people:
            m=ids==person
            if y[m].mean()>=90:high[m]=True
        assert len(np.unique(ids[high]))==68
        metric_difference.extend([abs(float((np.abs(p[high]-y[high])*weights[high]).sum()/weights[high].sum())-values['high90']['mae']),abs(float(((p[high]-y[high])*weights[high]).sum()/weights[high].sum())-values['high90']['bias'])])
    assert max(metric_difference)<1e-10
    cnn=z['old_CNN_bag']/3+z['CNN_seed20261101']/3+z['CNN_seed20261102']/3;assert abs(cnn-z['three_initialization_CNN']).max()<1e-10
    md=arrays(OLD/'mean-difference692/analysis/predictions.npz');old=arrays(read(OLD/'inner-bagging692/analysis/comparison.json')['predictions_path']);selection=read(OLD/'mean-difference692/analysis/selection.json')['folds']
    for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(z[k],old[k]);np.testing.assert_array_equal(z[k],md[k])
    result=np.zeros_like(y)
    for i in range(len(y)):
        s=next(s for s in selection if s['outer']==fold[i]);w,t=s['cnn_weight'],s['fraction'];left=w*cnn[i,0]+(1-w)*old['STT_bag'][i,0];right=w*cnn[i,1]+(1-w)*old['STT_bag'][i,1]
        difference=(left-right)*(1-t)+(md['bilateral_STT'][i,0]-md['bilateral_STT'][i,1])*t
        result[i]=[(left+right+difference)/2,(left+right-difference)/2]
    reconstruction=float(abs(result-z['three_initialization_blend']).max());assert reconstruction<1e-10
    assert abs(float((np.abs(z['baseline']-y)*weights).sum())-6.515816569178735)<1e-10
    delta=float(((np.abs(result-y)-np.abs(z['baseline']-y))*weights).sum());assert abs(delta-report['new_minus_baseline'])<1e-10
    assert report['target6_3_met']==(report['metrics']['three_initialization_blend']['mae']<=6.3)
    for p,digest in report['sources'].items():assert sha(p)==digest
    assert len(report['members'])==50 and report['maximum_replay_error']<1e-4
    out=dict(passed=True,person_weights_sum=float(weights.sum()),maximum_metric_difference=max(metric_difference),scalar_ensemble_reconstruction_max=reconstruction,all_50_members_checked=True,source_files_rehashed=len(report['sources']),comparison_sha256=sha(H/'comparison.json'),target6_3_met=report['target6_3_met'])
    (H/'checks/comparison-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
