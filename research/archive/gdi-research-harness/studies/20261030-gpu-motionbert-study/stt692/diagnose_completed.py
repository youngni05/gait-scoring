"""Post-hoc description of existing OOF only; no fitting or held-out tuning."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent
def main():
    paths={'single_cnn':S/'cnn-paper-input695/analysis/cnn_oof.npz','multi_cnn':S/'cnn-multiwindow695/analysis/cnn_oof.npz','single_stt':H/'full/analysis/unpacked/results/oof_predictions.npz'}
    data={}
    for name,path in paths.items():
        with np.load(path,allow_pickle=False) as z:data[name]={k:z[k] for k in z.files}
    d=data['multi_cnn'];y=d['y'];ids=d['person_id'];people=np.unique(ids)
    def avg(v):return np.array([v[ids==p].mean() for p in people])
    for z in data.values():
        for k in ('y','person_id','video_id','fold','original_row_index'):np.testing.assert_array_equal(z[k],d[k])
    pred={k:v['predictions'][:,0,:] for k,v in data.items()}
    errors={k:avg(abs(v-y)) for k,v in pred.items()};bias={k:avg(v-y) for k,v in pred.items()}
    means=avg(y.mean(1));asym=avg(abs(y[:,0]-y[:,1]));gdi_group=np.searchsorted([60,70,80,90],means,side='right');asym_group=np.searchsorted([5,10,20],asym,side='right')
    groups={}
    for kind,labels,names in [('mean_gdi',gdi_group,['below60','60to70','70to80','80to90','90plus']),('left_right_difference',asym_group,['below5','5to10','10to20','20plus'])]:
        groups[kind]=[]
        for i,name in enumerate(names):
            mask=labels==i
            groups[kind].append(dict(group=name,people=int(mask.sum()),mae={k:float(v[mask].mean()) for k,v in errors.items()},bias={k:float(v[mask].mean()) for k,v in bias.items()}))
        for k in pred:assert abs(sum(r['people']*r['mae'][k] for r in groups[kind])/len(people)-errors[k].mean())<1e-10
    bilateral={}
    for name,p in pred.items():
        truth=y[:,0]-y[:,1];delta=p[:,0]-p[:,1];strong=asym>=20
        bilateral[name]=dict(mean_score_mae=float(avg(abs(p.mean(1)-y.mean(1))).mean()),left_right_difference_mae=float(avg(abs(delta-truth)).mean()),true_abs_difference=float(asym.mean()),predicted_abs_difference=float(avg(abs(delta)).mean()),large_asymmetry_people=int(strong.sum()),large_asymmetry_true=float(asym[strong].mean()),large_asymmetry_predicted=float(avg(abs(delta))[strong].mean()))
    result=dict(people=len(people),records=len(y),overall_mae={k:float(v.mean()) for k,v in errors.items()},groups=groups,bilateral=bilateral,source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},real_optimizer_steps=0,note='Post-hoc development OOF descriptions, not causal evidence or independent final evaluation; no model or weights selected here.')
    out=H/'error-analysis/completed-models.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
