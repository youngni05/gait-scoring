"""Reuse all54 previous checkpoints/coefficients and exact forward functions, no labels in inference."""
import sys,json,hashlib,importlib.util,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'20261101-self-video-gdi';BIG=Path('D:/gait-research-data/D_P/self-video-test01-20261101')
spec=importlib.util.spec_from_file_location('previous_frozen_predict',str(OLD/'predict.py'));prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
np=prior.np;torch=prior.torch
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def predict(x,reference):
    assert x.dtype==np.float32 and x.shape[1:]==(2,124,12) and np.isfinite(x).all();out={}
    for member in reference['members']:
        family=member['family'];f=member['fold'];path=Path(member['checkpoint']);assert sha(path)==member['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=True);m=prior.new_model(family);m.load_state_dict(ck['model'],strict=True)
        pred=prior.forward(m,family,x,ck['lo'],ck['span']);a,b=member['calibration'];pred=pred*a+b
        key='bilateral_%d'%f if family=='bilateral' else '%s_%d_%s'%(family,f,member['tag']);out[key]=pred
    selections=read(prior.S/'mean-difference692/analysis/selection.json')['folds']
    for f,sf in enumerate(selections):
        for family in ['CNN','STT']:
            parts=[v for k,v in out.items() if k.startswith('%s_%d_'%(family,f))]
            assert len(parts)==(4 if family=='CNN' and f==0 else 5)
            out['%s_%d_bag'%(family,f)]=np.mean(parts,axis=0)
        base=sf['cnn_weight']*out['CNN_%d_bag'%f]+(1-sf['cnn_weight'])*out['STT_%d_bag'%f]
        common=base.mean(1);difference=(1-sf['fraction'])*(base[:,0]-base[:,1])+sf['fraction']*(out['bilateral_%d'%f][:,0]-out['bilateral_%d'%f][:,1])
        out['best_%d'%f]=np.column_stack([common+difference/2,common-difference/2])
    for model in ['best','CNN_refit','CNN_bag','STT_bag']:
        keys=['best_%d'%f for f in range(5)] if model=='best' else ['%s_%d_%s'%('CNN' if model.startswith('CNN') else 'STT',f,'refit' if model=='CNN_refit' else 'bag') for f in range(5)]
        out[model]=np.mean([out[k] for k in keys],axis=0)
    return out
def main():
    review=read(HERE/'prescore-review.json');assert review['reviewed_before_predictions']
    torch.set_num_threads(4);torch.manual_seed(20261101);torch.use_deterministic_algorithms(True)
    reference=read(OLD/'results.json');assert sha(OLD/'results.json')==read(OLD/'verification.json')['results_sha256']
    # Exact previously scored data: check every saved member, aggregate and final combination.
    old_big=Path('D:/gait-research-data/D_P/self-video-gdi-20261101');replay_x=np.load(old_big/'features.npz')['x'];replay=predict(replay_x,reference)
    with np.load(old_big/'predictions.npz') as z:replay_error=max(float(np.abs(v-z[k]).max()) for k,v in replay.items())
    assert replay_error<1e-4,replay_error;print('PREVIOUS_EXPERIMENT_REPLAY',replay_error,flush=True)
    windows=read(HERE/'windows.json');x=np.load(BIG/'features.npz')['x'];assert len(x)==len(windows);pred=predict(x,reference)
    np.savez_compressed(BIG/'predictions.npz',**pred)
    # Labels enter only the reporting stage after all predictions are fixed.
    labels=read(HERE/'labels.json');quality=read(HERE/'quality.json');rows=[]
    for source in read(HERE/'inventory.json'):
        number=source['number'];clip=source['id'];ix=[i for i,w in enumerate(windows) if w['id']==clip];q=next(r for r in quality if r['id']==clip)
        row=dict(number=number,id=clip,filename=source['filename'],condition='normal' if number in labels['normal_numbers'] else 'abnormal',status='scored' if ix else 'excluded',valid_windows=len(ix),seconds=source['seconds'])
        if ix:
            scores=pred['best'][ix].mean(1);fold_scores=[float(pred['best_%d'%f][ix].mean()) for f in range(5)]
            row.update(score=float(scores.mean()),left=float(pred['best'][ix,0].mean()),right=float(pred['best'][ix,1].mean()),window_min=float(scores.min()),window_max=float(scores.max()),fold_scores=fold_scores,fold_min=min(fold_scores),fold_max=max(fold_scores),models={name:float(pred[name][ix].mean()) for name in ['CNN_refit','CNN_bag','STT_bag']})
        else:row.update(score=None,reason=q.get('reason','no_eligible_input'))
        rows.append(row)
    normal=[r['score'] for r in rows if r['condition']=='normal' and r['score'] is not None];abnormal=[r['score'] for r in rows if r['condition']=='abnormal' and r['score'] is not None]
    group=dict(unit='video, no same-person matching or independence-based inference',normal_n=len(normal),abnormal_n=len(abnormal),normal_mean=float(np.mean(normal)) if normal else None,abnormal_mean=float(np.mean(abnormal)) if abnormal else None,normal_range=[min(normal),max(normal)] if normal else None,abnormal_range=[min(abnormal),max(abnormal)] if abnormal else None)
    if normal and abnormal:group['normal_minus_abnormal']=group['normal_mean']-group['abnormal_mean'];group['abnormal_above_lowest_normal_numbers']=[r['number'] for r in rows if r['condition']=='abnormal' and r['score'] is not None and r['score']>=min(normal)]
    report=dict(records=rows,group=group,previous_replay_max_error=replay_error,model_members=reference['members'],reference_results_sha256=sha(OLD/'results.json'),protocol_sha256=sha(HERE/'protocol.json'),features_sha256=sha(BIG/'features.npz'),predictions_sha256=sha(BIG/'predictions.npz'),windows_sha256=sha(HERE/'windows.json'),labels_used_for_inference=False,optimizer_steps=0)
    (HERE/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    fields=['number','condition','score','left','right','valid_windows','window_min','window_max','fold_min','fold_max','status']
    with (HERE/'video-scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(records=rows,group=group),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
