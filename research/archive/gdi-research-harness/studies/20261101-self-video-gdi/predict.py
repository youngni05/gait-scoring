"""Frozen best CNN/STT bagged ensemble; no fitting or user labels in inference."""
import sys,json,hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];S=HERE.parent/'20261030-gpu-motionbert-study'
BIG=Path('D:/gait-research-data/D_P/self-video-gdi-20261101');CACHE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study')
sys.path[:0]=[str(ROOT/'outputs/motionbert-smoke-20261015/deps'),str(S/'stt692/multiwindow/bundle'),str(S/'stt692/multiwindow/bundle/vendor')]
import numpy as np,torch
from model.Transformer import SpatioTemporalTransformer
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
CNN=module('cnn_frozen',S/'cnn-multiwindow695/bundle/model.py')
BIL=module('bil_frozen',S/'stt692/bilateral/bundle/bilateral.py')
def new_model(family):
    if family=='CNN':return CNN.GdiCNN()
    if family=='STT':return SpatioTemporalTransformer(num_frame=124,num_joints=4,in_chans=2,joint_embed_dim=12,num_heads=2,drop_rate=0.)
    return BIL.BilateralSTT()
def forward(model,family,x,lo,span):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(x),16):
            v=torch.from_numpy(x[start:start+16])
            if family=='CNN':value=model(CNN.channels(v).reshape(-1,12,124))[:,0]
            elif family=='STT':value=model(v[...,:8].reshape(-1,124,4,2))
            else:value=model(v[...,:8].reshape(-1,2,124,4,2))
            out.append((value*float(span)+float(lo)).reshape(-1,2).numpy())
    return np.concatenate(out).astype(np.float64)
def main():
    assert (HERE/'prescore-review.json').exists(),'Inspect pose overlays and freeze quality first'
    torch.set_num_threads(4);torch.manual_seed(20261101);torch.use_deterministic_algorithms(True)
    x=np.load(BIG/'features.npz')['x'];windows=read(HERE/'windows.json');assert x.shape==(len(windows),2,124,12) and np.isfinite(x).all()
    with np.load(S/'cnn-multiwindow695/bundle/data.npz') as z:
        source_x=z['x'];source_owner=z['window_record'];source_start=z['window_start'];source_fold=z['fold']
    source_lookup={(int(owner),int(start)):i for i,(owner,start) in enumerate(zip(source_owner,source_start))}
    allpred={};proofs=[]
    for f in range(5):
        for family in ['CNN','STT']:
            stack=[]
            for j in [None]+[j for j in range(5) if j!=f]:
                tag='refit' if j is None else 'inner%d'%j
                if j is not None:
                    p=read(CACHE/('inner-bagging692/cache/%s/fold%d_%s.json'%(family,f,tag)))
                    if p['excluded']:continue
                p=read(CACHE/('robust-aggregation692/cache/%s/fold%d_%s.json'%(family,f,tag)))
                ckpath=Path(p['checkpoint']);assert sha(ckpath)==p['checkpoint_sha256']
                ck=torch.load(ckpath,map_location='cpu',weights_only=True);assert ck['identity']['outer_fold']==f
                cache_path=CACHE/('robust-aggregation692/cache/%s/fold%d_%s.npz'%(family,f,tag));assert sha(cache_path)==p['cache_sha256']
                with np.load(cache_path) as z:
                    coef=z['calibration'][0].copy()
                    replay_ix=[source_lookup[int(owner),int(start)] for owner,start in zip(z['window_record'][:3],z['window_start'][:3])]
                    replay_expected=z['window_raw'][:3].copy()
                model=new_model(family);model.load_state_dict(ck['model'],strict=True)
                replay_error=float(np.abs(forward(model,family,source_x[replay_ix],ck['lo'],ck['span'])-replay_expected).max());assert replay_error<1e-4,(family,f,j,replay_error)
                raw=forward(model,family,x,ck['lo'],ck['span']);pred=raw*coef[0]+coef[1]
                allpred['%s_%d_%s'%(family,f,tag)]=pred;stack.append(pred)
                proofs.append(dict(family=family,fold=f,tag=tag,checkpoint=str(ckpath),checkpoint_sha256=sha(ckpath),calibration=coef.tolist(),cache=str(cache_path),cache_sha256=sha(cache_path),replay_max_error=replay_error))
            allpred['%s_%d_bag'%(family,f)]=np.mean(stack,axis=0)
        folder=S/('stt692/bilateral/analysis/unpacked/results/fold%d/refit'%f);done=read(folder/'fit_complete.json');ckpath=folder/'latest.pt'
        assert sha(ckpath)==done['checkpoint_sha256'];ck=torch.load(ckpath,map_location='cpu',weights_only=True)
        model=new_model('bilateral');model.load_state_dict(ck['model'],strict=True);coef=read(folder/'curve.json')[-1]['calibration']
        replay_record=int(np.flatnonzero(source_fold!=f)[0]);replay_x=source_x[source_owner==replay_record]
        with np.load(folder/('raw_predictions/epoch%04d.npz'%done['epochs'])) as z:replay_expected=z['train_raw'][0]
        replay_error=float(np.abs(forward(model,'bilateral',replay_x,ck['lo'],ck['span']).mean(0)-replay_expected).max());assert replay_error<1e-4,('bilateral',f,replay_error)
        bilateral=forward(model,'bilateral',x,ck['lo'],ck['span'])*coef[0]+coef[1]
        allpred['bilateral_%d'%f]=bilateral
        sf=read(S/'mean-difference692/analysis/selection.json')['folds'][f];assert sf['outer']==f
        base=sf['cnn_weight']*allpred['CNN_%d_bag'%f]+(1-sf['cnn_weight'])*allpred['STT_%d_bag'%f]
        avg=base.mean(1);diff=(1-sf['fraction'])*(base[:,0]-base[:,1])+sf['fraction']*(bilateral[:,0]-bilateral[:,1])
        allpred['best_%d'%f]=np.column_stack([avg+diff/2,avg-diff/2])
        proofs.append(dict(family='bilateral',fold=f,checkpoint=str(ckpath),checkpoint_sha256=sha(ckpath),calibration=coef,cnn_weight=sf['cnn_weight'],fraction=sf['fraction'],replay_max_error=replay_error))
        print('FOLD',f,'complete',flush=True)
    for name,key in [('best','best'),('CNN_refit','CNN'),('CNN_bag','CNN'),('STT_bag','STT')]:
        keys=['best_%d'%f for f in range(5)] if name=='best' else ['%s_%d_%s'%(key,f,'refit' if name=='CNN_refit' else 'bag') for f in range(5)]
        allpred[name]=np.mean([allpred[k] for k in keys],axis=0)
    np.savez_compressed(BIG/'predictions.npz',**allpred)
    summary={}
    for kind in ['primary','straight']:
        summary[kind]={}
        for model in ['best','CNN_refit','CNN_bag','STT_bag']:
            groups={clip:[i for i,w in enumerate(windows) if w['clip']==clip and w['kind']==kind] for clip in ['clip0','clip1']}
            stats={clip:dict(n_windows=len(ix),left_right=allpred[model][ix].mean(0).tolist(),mean=float(allpred[model][ix].mean()),window_means=allpred[model][ix].mean(1).tolist(),starts=[windows[i]['start'] for i in ix]) for clip,ix in groups.items()}
            stats['normal_minus_abnormal']=stats['clip1']['mean']-stats['clip0']['mean']
            foldkeys=['best_%d'%f for f in range(5)] if model=='best' else ['%s_%d_%s'%('CNN' if model.startswith('CNN') else 'STT',f,'refit' if model=='CNN_refit' else 'bag') for f in range(5)]
            stats['fold_differences']=[float(allpred[k][groups['clip1']].mean()-allpred[k][groups['clip0']].mean()) for k in foldkeys]
            summary[kind][model]=stats
    report=dict(summary=summary,windows_sha256=sha(HERE/'windows.json'),protocol_sha256=sha(HERE/'protocol.json'),features_sha256=sha(BIG/'features.npz'),predictions_sha256=sha(BIG/'predictions.npz'),members=proofs,optimizer_steps=0,true_GDI_available=False)
    (HERE/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
