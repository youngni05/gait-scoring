"""Read-only complete-run audit and ten final CPU64 replays. Run AFTER receipt/SHA checks."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import run as r
from audit import audit_fit

def subgroup(y,p,people,mask):
    errors=p-y;mae=[];bias=[];n=0
    for person in np.unique(people):
        m=mask[people==person];e=errors[people==person][m]
        if len(e):mae.append(float(abs(e).mean()));bias.append(float(e.mean()));n+=len(e)
    return dict(people=len(mae),side_observations=n,mae=float(np.mean(mae)) if mae else None,bias=float(np.mean(bias)) if bias else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--received',type=Path,required=True);ap.add_argument('--receipt',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    receipt=r.load(a.receipt);assert receipt.get('passed') and receipt.get('server_local_sha256_matched'), 'Verify actual downloads first'
    r.setup(7,'cpu');d=r.data();results={};errors={};replays=[]
    for variant in ['scratch','synth']:
        c=r.config_for(variant,'main')
        out=a.received/'runs/main'/variant;pred=np.full_like(d['y'],np.nan);raw_all=pred.copy();seen=np.zeros(len(pred),int)
        for f in range(5):
            folder=out/('fold%d'%f);tr,_,te=r.split(d,f)
            histories=[audit_fit(r,d,c,f,j,folder/('inner%d'%j),c['max_epochs'],'inner_epoch_selection') for j in range(5) if j!=f]
            epoch,means=r.select_epoch(histories);sel=r.load(folder/'selection.json')
            assert epoch==sel['selected_epoch'];np.testing.assert_allclose(means,sel['inner_mean_person_mae_by_epoch'],rtol=0,atol=1e-10)
            audit_fit(r,d,c,f,None,folder/'refit',epoch,'outer_refit')
            done=r.load(folder/'complete.json');assert r.sha(folder/'predictions.npz')==done['prediction_sha256']
            ck=torch.load(folder/'refit/latest.pt',map_location='cpu',weights_only=True)
            model=r.GdiCNN(c['dropout']).double();model.load_state_dict(ck['model']);model.eval()
            x=d['x'][te];parts=[]
            with torch.no_grad():
                for start in range(0,len(x.values),128):
                    t=r.channels(torch.from_numpy(x.values[start:start+128])).reshape(-1,12,124).double()
                    p=model(t)[:,0]*ck['span']+ck['lo'];parts.append(p.reshape(-1,2).numpy())
            raw=x.aggregate(np.concatenate(parts))
            with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                ix=np.flatnonzero(te);np.testing.assert_array_equal(ix,z['row_index'])
                for k in ['person_id','video_id','fold','cohort','y']:np.testing.assert_array_equal(z[k],d[k][ix])
                ref=z['predictions_raw'][:,0];corrected=r.corrected(raw,z['calibration'])
                err=max(float(np.max(abs(raw-ref))),float(np.max(abs(corrected-z['predictions'][:,0]))))
                assert err<1e-4,err
                pred[ix]=z['predictions'][:,0];raw_all[ix]=ref;seen[ix]+=1
                replays.append(dict(variant=variant,fold=f,error=err,tolerance=1e-4,checkpoint_sha256=r.sha(folder/'refit/latest.pt')))
        assert (seen==1).all() and np.isfinite(pred).all()
        with np.load(out/'oof_predictions.npz') as z:np.testing.assert_array_equal(z['predictions'][:,0],pred)
        q=dict(overall=r.metrics(d['y'],pred,d['person_id']),raw=r.metrics(d['y'],raw_all,d['person_id']),subgroups={})
        for threshold in [90,100]:
            mask=d['y']>=threshold;q['subgroups']['observation_ge%d'%threshold]=subgroup(d['y'],pred,d['person_id'],mask)
            selected=[p for p in np.unique(d['person_id']) if d['y'][d['person_id']==p].mean()>=threshold]
            mask=np.repeat(np.isin(d['person_id'],selected)[:,None],2,axis=1)
            q['subgroups']['person_mean_ge%d'%threshold]=subgroup(d['y'],pred,d['person_id'],mask)
        q['subgroups']['observation_lt70']=subgroup(d['y'],pred,d['person_id'],d['y']<70)
        results[variant]=q
        errors[variant]=np.array([abs(pred[d['person_id']==p]-d['y'][d['person_id']==p]).mean() for p in np.unique(d['person_id'])])
    delta=errors['synth']-errors['scratch'];rng=np.random.default_rng(20261031)
    bootstrap=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)])
    import stem
    external_proofs=[stem.audit_external(p,r.config(),r,'cpu') for p in sorted((r.ROOT/'runs/pretrain/main').glob('seed*'))]
    assert len(external_proofs)==25
    r.save(a.output,dict(passed=True,receipt_sha256=r.sha(a.receipt),fits=50,external_pretraining=external_proofs,final_replays=replays,results=results,
        synth_minus_scratch_mae=float(delta.mean()),person_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),independent_clinical_validation=False))
if __name__=='__main__':main()
