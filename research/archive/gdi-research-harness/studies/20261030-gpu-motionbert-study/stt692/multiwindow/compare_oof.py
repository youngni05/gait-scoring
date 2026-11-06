"""Matched-person OOF comparison; descriptive only, no fitting or model choice."""
from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
z=arrays(H/'analysis/unpacked/results/oof_predictions.npz')
ids=z['person_id'];people=np.unique(ids);y=z['y'];p=z['predictions'][:,0];raw=z['predictions_raw'][:,0]
assert len(people)==692 and len(ids)==1061 and len(set(z['video_id']))==1061
def person_errors(pred):return np.array([abs(pred[ids==q]-y[ids==q]).mean() for q in people])
models={'multiwindow_STT':p,'multiwindow_STT_raw':raw}
sources={'multiwindow_CNN':H.parent.parent/'cnn-multiwindow695/analysis/cnn_oof.npz',
 'single_CNN':H.parent.parent/'cnn-paper-input695/analysis/cnn_oof.npz',
 'single_STT':H.parent/'full/analysis/unpacked/results/oof_predictions.npz'}
for name,path in sources.items():
    v=arrays(path);lookup={q:i for i,q in enumerate(v['video_id'])};ix=np.array([lookup[q] for q in z['video_id']])
    assert len(lookup)==1061
    for key in ['person_id','y','fold']:np.testing.assert_array_equal(v[key][ix],z[key])
    models[name]=v['predictions'][ix,0]
errors={k:person_errors(v) for k,v in models.items()}
paired={}
for name in sources:
    delta=errors['multiwindow_STT']-errors[name];rng=np.random.default_rng(20261031)
    boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(10000)])
    paired[name]=dict(stt_minus_reference_mae=float(delta.mean()),relative_change_pct=float(delta.mean()/errors[name].mean()*100),
        stt_better_people=int((delta<0).sum()),stt_worse_people=int((delta>0).sum()),
        paired_person_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        note='Descriptive fixed OOF paired bootstrap; excludes training-seed and model-selection uncertainty; not independent clinical validation')
group_mean=np.array([y[ids==q].mean() for q in people]);groups={}
for label,mask in [('person_mean_GDI_below60',group_mean<60),('person_mean_GDI_60to90',(group_mean>=60)&(group_mean<90)),('person_mean_GDI_atleast90',group_mean>=90)]:
    groups[label]=dict(people=int(mask.sum()),models={})
    for name,pred in models.items():
        biases=np.array([(pred[ids==q]-y[ids==q]).mean() for q in people])
        groups[label]['models'][name]=dict(mae=float(errors[name][mask].mean()),bias=float(biases[mask].mean()))
result=dict(people=692,records=1061,matched_video_person_label_fold=True,person_equal_mae={k:float(v.mean()) for k,v in errors.items()},paired=paired,
    descriptive_groups=groups,protected_final_data_opened=False,model_fitting_steps=0)
(H/'analysis/paired_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='descriptive_groups'}))
