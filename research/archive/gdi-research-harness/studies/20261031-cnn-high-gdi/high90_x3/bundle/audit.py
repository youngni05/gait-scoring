"""Training-only weights, baseline invariance, pilot checkpoint and final fit audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import run
from weighting import example_weights, record_side_weights

def check_weights(d, c):
    rows = []
    for outer in range(5):
        for inner in [None]+[i for i in range(5) if i != outer]:
            tr, _, _ = run.split(d, outer, inner)
            x, y = d['x'][tr], d['y'][tr]
            w, audit = example_weights(x, y, c['score_weighting'])
            baseline, _ = example_weights(x, y, 'baseline')
            np.testing.assert_array_equal(baseline, x.side_weights())
            np.testing.assert_allclose(w.mean(), 1., rtol=0, atol=2e-7)
            # Summing all windows must give exactly the intended relative record-side contributions.
            rs, _ = record_side_weights(y, c['score_weighting'])
            total = np.zeros_like(y)
            np.add.at(total, x.owner, w.reshape(-1,2))
            np.testing.assert_allclose(total/len(w), rs/rs.size, rtol=2e-7, atol=1e-10)
            rows.append(dict(outer=outer, inner=inner, **audit))
    # Boundary and rare-bin direction tests, including an empty bin.
    toy = np.array([[59.,69.],[79.,89.],[90.,100.]])
    a, _ = record_side_weights(toy, 'high90_x3')
    np.testing.assert_array_equal(a[toy>=90]/a[0,0], np.full(2,3.))
    b, _ = record_side_weights(np.array([[50.,50.],[50.,95.]]), 'rare_sqrt_cap4')
    assert b[1,1] > b[0,0]
    return rows

def replay_fit(d, c, folder, outer, inner, epochs, purpose, device):
    tr, va, _ = run.split(d, outer, inner)
    identity = run.identity_for(d, c, outer, inner, epochs, purpose)
    assert run.load(folder/'identity.json') == identity
    ck = torch.load(folder/'latest.pt', map_location='cpu', weights_only=True)
    assert ck['identity'] == identity and len(ck['history']) == epochs
    done = run.load(folder/'fit_complete.json')
    assert done['checkpoint_sha256'] == run.sha(folder/'latest.pt')
    assert done['curve_sha256'] == run.sha(folder/'curve.json')
    assert done['optimizer_steps'] == epochs*int(np.ceil(2*len(d['x'][tr].values)/c['batch_size']))
    _, weighting = example_weights(d['x'][tr], d['y'][tr], c['score_weighting'])
    assert run.load(folder/'weighting.json') == weighting
    lo, span = run.scale_from_train(d['y'][tr])
    assert torch.equal(lo,ck['lo']) and torch.equal(span,ck['span'])
    model = run.GdiCNN(c['dropout']).to(device)
    model.load_state_dict(ck['model'], strict=True)
    raw = run.predict(model,d['x'][tr],lo,span,device)
    coef = run.calibrate(raw,d['y'][tr])
    row = ck['history'][-1]
    np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-4)
    for mask, prefix in ((tr,'train'), (va,'validation')):
        if not mask.any(): continue
        p = raw if prefix=='train' else run.predict(model,d['x'][mask],lo,span,device)
        for tag, values in (('',run.corrected(p,coef)), ('_raw',p)):
            metric = run.person_mae(d['y'][mask],values,d['person_id'][mask])
            np.testing.assert_allclose(metric,row[prefix+tag+'_person_mae'],rtol=0,atol=1e-4)
    return model,lo,span,coef

def main():
    p=argparse.ArgumentParser(); p.add_argument('command', choices=['weights','pilot','final'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=run.config();d=run.data();run.setup(c['seed'],a.device)
    if a.command=='weights':
        rows=check_weights(d,c)
        run.save(a.output/'weight_check.json',{'passed':True,'fits':rows,'baseline_bitwise_equal':True})
    elif a.command=='pilot':
        replay_fit(d,c,a.output/'pilot',0,1,c['pilot_epochs'],'pilot_not_performance',a.device)
        run.save(a.output/('pilot_replay_'+a.device+'.json'),{'passed':True,'absolute_tolerance':1e-4})
    else:
        run.summarize(d,c,a.output)
        for f in range(5):
            dest=a.output/('fold%d'%f);done=run.load(dest/'complete.json')
            for inner in [i for i in range(5) if i!=f]:
                fit=dest/('inner%d'%inner);info=run.load(fit/'fit_complete.json')
                assert info['checkpoint_sha256']==run.sha(fit/'latest.pt')
                _, wa=example_weights(d['x'][run.split(d,f,inner)[0]],d['y'][run.split(d,f,inner)[0]],c['score_weighting'])
                assert run.load(fit/'weighting.json')==wa
            model,lo,span,coef=replay_fit(d,c,dest/'refit',f,None,done['selected_epoch'],'outer_refit',a.device)
            te=run.split(d,f)[2]
            raw=run.predict(model,d['x'][te],lo,span,a.device)
            with np.load(dest/'predictions.npz') as z:
                np.testing.assert_allclose(raw,z['predictions_raw'][:,0],rtol=0,atol=1e-4)
                np.testing.assert_allclose(run.corrected(raw,coef),z['predictions'][:,0],rtol=0,atol=1e-4)
        run.save(a.output/('final_replay_'+a.device+'.json'),{'passed':True,'folds':5,'fits':25,'absolute_tolerance':1e-4})
    print(json.dumps({'passed':True,'command':a.command,'device':a.device}))
if __name__=='__main__':main()
