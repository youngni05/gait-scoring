"""Read existing CNN refit checkpoints; export predictions without optimization.
Run beside the ORIGINAL server bundle/results; writes only a new output folder.
Python 3.8+, NumPy and PyTorch 2.4+. CPU default; no GPU allocation required.
"""
import argparse,importlib.util,json,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=ap.parse_args();bundle=args.bundle.resolve();results=args.results.resolve();out=args.out.resolve()
    if out.exists():raise ValueError('Output must be a new folder')
    sys.path.insert(0,str(bundle))
    spec=importlib.util.spec_from_file_location('original_cnn_runner',bundle/'run.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    np,torch=c.np,c.torch;d=c.data();config=c.config()
    # Check every checkpoint before creating output or performing inference.
    proofs=[]
    for f in range(5):
        folder=results/('fold%d'%f);done=c.load(folder/'complete.json');fit=c.load(folder/'refit/fit_complete.json')
        ckpath=folder/'refit/latest.pt'
        c.require(c.sha(ckpath)==done['checkpoint_sha256']==fit['checkpoint_sha256'],'Checkpoint hash mismatch')
        c.require(c.sha(folder/'predictions.npz')==done['prediction_sha256'],'Held predictions hash mismatch')
        c.require(done['identity']==c.identity_for(d,config,f,None,done['selected_epoch'],'outer_refit'),'Run identity mismatch')
        proofs.append((folder,done,ckpath))
    c.setup(config['seed'],args.device);out.mkdir(parents=True)
    audits=[]
    for f,(folder,done,ckpath) in enumerate(proofs):
        ck=torch.load(ckpath,map_location=args.device,weights_only=True)
        c.require(ck['identity']==done['identity'],'Loaded checkpoint identity mismatch')
        c.require(len(ck['history'])==done['selected_epoch'],'Checkpoint epoch mismatch')
        model=c.GdiCNN(config['dropout']).to(args.device);model.load_state_dict(ck['model'],strict=True)
        tr,_,te=c.split(d,f);lo,span=c.scale_from_train(d['y'][tr])
        c.require(torch.equal(ck['lo'].cpu(),lo) and torch.equal(ck['span'].cpu(),span),'Scaling mismatch')
        before=c.sha(ckpath)
        pred=c.predict(model,c.channels(torch.from_numpy(d['x'])),lo,span,args.device)
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(np.flatnonzero(te),z['row_index'])
            diff=float(np.max(np.abs(pred[te]-z['predictions'][:,0])))
            np.testing.assert_allclose(pred[te],z['predictions'][:,0],atol=1e-4,rtol=0)
        c.require(c.sha(ckpath)==before,'Checkpoint changed during inference')
        path=out/('fold%d.npz'%f)
        np.savez_compressed(path,train_predictions=pred[tr],train_indices=np.flatnonzero(tr),
            held_predictions=pred[te],held_indices=np.flatnonzero(te),
            person_id=d['person_id'],video_id=d['video_id'],fold=d['fold'],y=d['y'])
        audits.append(dict(fold=f,source_checkpoint_sha256=before,source_complete_sha256=c.sha(folder/'complete.json'),
            export_sha256=c.sha(path),max_held_reproduction_difference=diff,train_people=len(set(d['person_id'][tr])),held_people=len(set(d['person_id'][te]))))
    c.save(out/'complete.json',dict(status='predictions_exported_no_training',bundle_sha256=c.sha(bundle/'manifest.json'),
        dataset_sha256=c.sha(bundle/'data.npz'),device=args.device,optimizer_steps=0,folds=audits))
    print(json.dumps(dict(output=str(out),optimizer_steps=0,folds=5)))

if __name__=='__main__':main()
