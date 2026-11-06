"""Supplemental local CPU64 model replay, separate from the unchanged strict audit."""
from pathlib import Path
import argparse,hashlib,json,os,sys
import numpy as np
H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--received',type=Path,required=True);ap.add_argument('--seed',type=int,choices=[20261101,20261102],required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    root=a.received.resolve();sys.path.insert(0,str(root));import torch;import run as r
    assert sha(root/'manifest.json')==sha(H/('seed%d/manifest.json'%a.seed));r.verify_manifest();r.setup(7,'cpu');d=r.data();c=r.config();assert c['initialization_seed']==a.seed
    proof=[]
    for f in range(5):
        tr,_,te=r.split(d,f);folder=root/('runs/main/fold%d'%f);done=r.load(folder/'complete.json');fit=r.load(folder/'refit/fit_complete.json');path=folder/'refit/latest.pt'
        assert sha(path)==done['checkpoint_sha256']==fit['checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=True)
        assert ck['identity']==fit['identity']==done['identity']==r.identity_for(d,c,f,None,done['selected_epoch'],'outer_refit')
        assert ck['history']==r.load(folder/'refit/curve.json');lo,span=r.scale_from_train(d['y'][tr]);assert torch.equal(lo,ck['lo']) and torch.equal(span,ck['span'])
        model=r.GdiCNN(c['dropout']).double();model.load_state_dict(ck['model']);model.eval()
        def infer(mask):
            x=d['x'][mask];parts=[]
            with torch.inference_mode():
                for start in range(0,len(x.values),128):
                    t=r.channels(torch.from_numpy(x.values[start:start+128])).reshape(-1,12,124).double();v=model(t)[:,0]*float(span)+float(lo);parts.append(v.numpy().reshape(-1,2))
            return x.aggregate(np.concatenate(parts))
        raw=infer(te);train_raw=infer(tr);coef=np.array(done['calibration']);np.testing.assert_allclose(coef,ck['history'][-1]['calibration'],rtol=0,atol=1e-10)
        assert sha(folder/'predictions.npz')==done['prediction_sha256']
        with np.load(folder/'predictions.npz') as z:
            for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(z[k],d[k][te])
            error=max(float(abs(raw-z['predictions_raw'][:,0]).max()),float(abs(r.corrected(raw,coef)-z['predictions'][:,0]).max()))
        epoch=done['selected_epoch'];rawpath=folder/'refit/raw_predictions'/('epoch%04d.npz'%epoch);assert sha(rawpath)==ck['history'][-1]['raw_predictions_sha256']
        with np.load(rawpath) as z:error=max(error,float(abs(train_raw-z['train_raw']).max()),float(abs(r.corrected(train_raw,coef)-r.corrected(z['train_raw'],coef)).max()))
        assert error<1e-4,(f,error)
        proof.append(dict(fold=f,maximum_prediction_error=error,checkpoint_sha256=sha(path),tolerance=1e-4));print(json.dumps(proof[-1]),flush=True)
    result=dict(passed=True,scope='supplemental local model replay only; does not replace or pass the original local coefficient audit',seed=a.seed,final_cpu64_replays=proof,maximum_prediction_error=max(v['maximum_prediction_error'] for v in proof),torch=str(torch.__version__),numpy=np.__version__,tolerance=1e-4,manifest_sha256=sha(root/'manifest.json'))
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
