"""Read-only complete-run audit, after separate receipt SHA256 verification."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import run as r
from ordinal import OrdinalCNN,targets
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert r.sha(a.receipt)==a.receipt_sha256
    # The caller's receipt must bind each extracted original result/model file.
    receipt=r.read(a.receipt);assert receipt['passed'] and receipt['server_local_sha256_matched']
    assert receipt['manifest_sha256']==r.sha(r.H/'manifest.json')
    for name,digest in receipt['result_files'].items():
        path=(a.results/name).resolve();assert a.results.resolve() in path.parents and r.sha(path)==digest,name
    d=r.data();c=r.read(r.H/'protocol.json');summary=r.read(a.results/'summary.json');r.setup(c['seed'],'cpu')
    fit_count=0;replays=[];metrics={};selected=[]
    for group in c['groups']:
        width,mode=r.spec(group);seen=np.zeros(len(d['y']),int);raw_all=np.empty_like(d['y']);prob_all=None
        for f in range(5):
            dest=a.results/group/('fold%d'%f);hist={};lambdas=c['joint_lambdas'] if group.startswith('joint') else [1.0]
            for lam in lambdas:
                hs=[]
                for j in range(5):
                    if j==f:continue
                    hs.append(r.audit(d,c,group,f,j,lam,c['max_epochs'],dest/('lambda'+str(lam))/('inner%d'%j)));fit_count+=1
                hist[str(lam)]=hs
            chosen,candidates=r.choose(hist,group,lambdas);selection=r.read(dest/'selection.json');assert not selection['outer_used']
            assert chosen==selection['chosen'] and candidates==selection['candidates']
            folder=dest/'refit';r.audit(d,c,group,f,None,chosen['lam'],chosen['epoch'],folder);fit_count+=1
            ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True);m=OrdinalCNN(width,mode);m.load_state_dict(ck['model'])
            _,_,te=r.split(d,f);z=r.arrays(dest/'predictions.npz');done=r.read(dest/'complete.json')
            assert done['chosen']==chosen and done['checkpoint_sha256']==r.sha(folder/'latest.pt')
            assert done['predictions_sha256']==r.sha(dest/'predictions.npz') and done['selection_sha256']==r.sha(dest/'selection.json')
            np.testing.assert_array_equal(te,z['row_index'])
            for key in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(z[key],d[key][te])
            raw,prob=r.predict(m,d,te,ck['lo'],ck['span'],'cpu',torch.float64);errors={}
            for key,val in [('raw',raw),('prob',prob)]:
                if val.size:errors[key]=float(abs(val-z[key]).max())
            proof=dict(group=group,fold=f,errors=errors,passed=max(errors.values())<1e-4,checkpoint_sha256=r.sha(folder/'latest.pt'))
            r.save(a.output/(group+'-fold%d-replay.json'%f),proof);assert proof['passed'],proof;replays.append(proof)
            seen[te]+=1
            if raw.size:raw_all[te]=z['raw']
            if prob.size:
                if prob_all is None:prob_all=np.full((len(d['y']),2,prob.shape[-1]),np.nan)
                prob_all[te]=z['prob']
            selected.append(dict(group=group,fold=f,**chosen))
        assert (seen==1).all();oof=r.arrays(a.results/group/'oof.npz')
        assert r.sha(a.results/group/'oof.npz')==summary['groups'][group]['oof_sha256']
        for key in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(oof[key],d[key])
        if mode=='ordinal':raw_all=np.empty((0,2))
        if prob_all is None:prob_all=np.empty((0,2,0))
        np.testing.assert_array_equal(raw_all,oof['raw']);np.testing.assert_array_equal(prob_all,oof['prob'])
        got=r.metrics(d['y'],raw_all,prob_all,d['person_id'],width)
        for key,val in got.items():assert abs(val-summary['groups'][group]['metrics'][key])<1e-10
        # Independent row weights and hard-bin accuracy/MAE.
        _,inv,count=np.unique(d['person_id'],return_inverse=True,return_counts=True);weights=1/(len(count)*count[inv])
        if raw_all.size:assert abs(float(np.sum(weights*abs(raw_all-d['y']).mean(1)))-got['mae'])<1e-10
        if prob_all.size:
            cls=np.clip(np.floor(d['y']/width).astype(int)+1,0,160//width+1)
            assert abs(float(np.sum(weights*(prob_all.argmax(-1)==cls).mean(1)))-got['accuracy'])<1e-10
        metrics[group]=got
    assert fit_count==165 and len(replays)==25
    report=dict(passed=True,fits=165,outer_replays=25,people=692,records=1061,metrics=metrics,selected=selected,replays=replays,
        receipt_sha256=a.receipt_sha256,protected_final_opened=False,local_optimizer_steps=0,raw_probabilities_uncalibrated=True)
    r.save(a.output/'verification.json',report);print(json.dumps(report))
if __name__=='__main__':main()
