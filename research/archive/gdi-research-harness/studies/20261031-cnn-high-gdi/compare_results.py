"""Independent held-out comparison and frozen LJMU checkpoint inference after retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20261030-gpu-motionbert-study/cnn-multiwindow695'
NORMAL=HERE.parent/'20261031-ljmu-normal-cnn'
sys.path.insert(0,str(BASE/'bundle'))
from model import GdiCNN, channels
sys.path.insert(0,str(HERE))
from weighting import record_side_weights

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def macro(values,people,mask=None):
    if mask is None:mask=np.ones_like(values,dtype=bool)
    return float(np.mean([values[(people==p)[:,None]&mask].mean() for p in np.unique(people) if (((people==p)[:,None])&mask).any()]))
def groups(y):
    return {'all':np.ones_like(y,dtype=bool),'below70':y<70,'90plus':y>=90,'100plus':y>=100,
            **{str(lo)+'-'+str(hi):(y>=lo)&(y<hi) for lo,hi in [(0,60),(60,70),(70,80),(80,90),(90,100),(100,200)]}}
def scores(y,p,people):
    result={}
    for name,mask in groups(y).items():
        if not mask.any():continue
        result[name]={'people':len(np.unique(people[mask.any(axis=1)])),'sides':int(mask.sum()),
                      'mae':macro(np.abs(p-y),people,mask),'bias':macro(p-y,people,mask),
                      'actual_mean':macro(y,people,mask),'predicted_mean':macro(p,people,mask)}
    return result
def bootstrap(y,p,b,people):
    rng=np.random.default_rng(20261031);result={}
    for name,mask in groups(y).items():
        if name not in ('all','below70','90plus','100plus'):continue
        diffs=np.array([(np.abs(p-y)-np.abs(b-y))[(people==pid)[:,None]&mask].mean()
                        for pid in np.unique(people) if ((people==pid)[:,None]&mask).any()])
        estimates=np.mean(diffs[rng.integers(0,len(diffs),size=(5000,len(diffs)))],axis=1)
        result[name]={'candidate_minus_baseline_mae':float(diffs.mean()),'person_bootstrap_95ci':np.quantile(estimates,[.025,.975]).tolist(),
                      'interpretation':'descriptive development-cohort comparison, not independent confirmatory evaluation'}
    return result
def predict(model,d,lo,span):
    output=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(d['x']),128):
            x=channels(torch.from_numpy(d['x'][i:i+128])).reshape(-1,12,124)
            output.append((model(x)[:,0]*span+lo).reshape(-1,2).numpy())
    windows=np.concatenate(output);owner=d['window_record']
    sums=np.zeros((len(d['person_id']),2),dtype=np.float64)
    np.add.at(sums,owner,windows)
    return sums/np.bincount(owner)[:,None]
def audit_and_infer(mode,d,received):
    config=read(HERE/mode/'bundle/protocol.json');manifestsha=sha(HERE/mode/'bundle/manifest.json')
    normal=load(NORMAL/'input/normal_cnn_v2/prepared_normal/prepared.npz')
    normal_preds=[];replayed=np.full_like(d['y'],np.nan);audit=[]
    for f in range(5):
        fold=received/('fold%d'%f);complete=read(fold/'complete.json');te=d['fold']==f;tr=~te
        histories=[]
        for inner in [i for i in range(5) if i!=f]+[None]:
            directory=fold/('refit' if inner is None else 'inner%d'%inner)
            train=tr if inner is None else tr&(d['fold']!=inner)
            valid=np.zeros_like(tr) if inner is None else d['fold']==inner
            epochs=complete['selected_epoch'] if inner is None else config['max_epochs']
            identity=read(directory/'identity.json')
            expected={'bundle_sha256':manifestsha,'purpose':'outer_refit' if inner is None else 'inner_epoch_selection',
                      'outer_fold':f,'inner_fold':inner,'epochs':epochs,'fit_seed':config['seed']+f*100+(50 if inner is None else inner),
                      'train_people':np.unique(d['person_id'][train]).tolist(),'validation_people':np.unique(d['person_id'][valid]).tolist(),
                      'outer_evaluation_people':np.unique(d['person_id'][te]).tolist()}
            assert identity==expected
            done=read(directory/'fit_complete.json');assert done['identity']==expected and done['epochs']==epochs
            assert sha(directory/'curve.json')==done['curve_sha256']
            nwin=int(np.isin(d['window_record'],np.flatnonzero(train)).sum())
            assert done['optimizer_steps']==epochs*int(np.ceil(2*nwin/config['batch_size']))
            _,weight=record_side_weights(d['y'][train],mode)
            assert read(directory/'weighting.json')==weight
            curve=read(directory/'curve.json');assert len(curve)==epochs
            if inner is not None:histories.append(curve)
        means=np.average([[r['validation_person_mae'] for r in c] for c in histories],axis=0,
                         weights=[c[0]['validation_people'] for c in histories])
        chosen=int(np.argmin(means))+1;assert chosen==complete['selected_epoch']
        np.testing.assert_allclose(means,read(fold/'selection.json')['inner_mean_person_mae_by_epoch'],rtol=0,atol=1e-10)
        assert sha(fold/'selection.json')==complete['selection_sha256']
        assert sha(fold/'predictions.npz')==complete['prediction_sha256']
        checkpoint=fold/'refit/latest.pt';assert sha(checkpoint)==complete['checkpoint_sha256']
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True);assert ck['identity']==identity
        assert float(ck['lo'])==float(np.float32(d['y'][tr].min()))
        assert float(ck['span'])==float(np.float32(d['y'][tr].max()-d['y'][tr].min()))
        model=GdiCNN(config['dropout']);model.load_state_dict(ck['model'],strict=True)
        raw=predict(model,d,ck['lo'],ck['span'])
        coef=np.linalg.lstsq(np.column_stack([raw[tr].ravel(),np.ones(raw[tr].size)]),d['y'][tr].ravel(),rcond=None)[0]
        np.testing.assert_allclose(coef,complete['calibration'],rtol=0,atol=1e-4)
        # Stored calibration keeps the defined exported predictor; recomputation separately verifies fit provenance.
        coef=np.asarray(complete['calibration']);pred=raw*coef[0]+coef[1]
        saved=load(fold/'predictions.npz')
        for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(saved[key],d[key][te])
        np.testing.assert_allclose(raw[te],saved['predictions_raw'][:,0],rtol=0,atol=1e-4)
        np.testing.assert_allclose(pred[te],saved['predictions'][:,0],rtol=0,atol=1e-4)
        replayed[te]=pred[te]
        nr=predict(model,normal,ck['lo'],ck['span']);normal_preds.append(nr*coef[0]+coef[1])
        audit.append({'fold':f,'selected_epoch':chosen,'replay_max_error':float(np.abs(pred[te]-saved['predictions'][:,0]).max()),'checkpoint_sha256':sha(checkpoint)})
        print(json.dumps({'mode':mode,'verified_fold':f}),flush=True)
    normal_preds=np.stack(normal_preds)
    np.savez_compressed(HERE/mode/'analysis/ljmu_predictions.npz',predictions=normal_preds,person_id=normal['person_id'])
    avg=normal_preds.mean(axis=0);person=np.array([avg[normal['person_id']==p].mean() for p in np.unique(normal['person_id'])])
    return audit,{'people':len(person),'mean':float(person.mean()),'sd':float(person.std(ddof=1)),'min':float(person.min()),'max':float(person.max()),
                  'note':'no measured GDI; score shift is not accuracy improvement'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','high90_x3','rare_sqrt_cap4'],required=True);p.add_argument('--received',type=Path);a=p.parse_args()
    torch.set_num_threads(4)
    d=load(BASE/'bundle/data.npz');base=load(BASE/'analysis/cnn_oof.npz')
    for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(base[key],d[key])
    bp=base['predictions'][:,0]
    report={'baseline':scores(d['y'],bp,d['person_id']),'baseline_oof_sha256':sha(BASE/'analysis/cnn_oof.npz')}
    if a.mode!='baseline':
        out=HERE/a.mode/'analysis';out.mkdir(exist_ok=True)
        received=(a.received or HERE/a.mode/'received')/'results';new=load(received/'oof_predictions.npz')
        for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(new[key],d[key])
        for f in range(5):
            saved=load(received/('fold%d'%f)/'predictions.npz')
            np.testing.assert_array_equal(new['predictions'][d['fold']==f],saved['predictions'])
            np.testing.assert_array_equal(new['predictions_raw'][d['fold']==f],saved['predictions_raw'])
        audit,normal=audit_and_infer(a.mode,d,received)
        pred=new['predictions'][:,0]
        report.update(candidate=scores(d['y'],pred,d['person_id']),raw_candidate=scores(d['y'],new['predictions_raw'][:,0],d['person_id']),
                      paired_mae_comparison=bootstrap(d['y'],pred,bp,d['person_id']),fold_replay=audit,ljmu=normal,
                      mode=a.mode,independent_final_evaluation=False,verified_fits=25)
        save(out/'comparison.json',report)
    else:save(HERE/'baseline_comparison.json',report)
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
