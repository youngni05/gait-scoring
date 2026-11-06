"""Audit all epoch metrics and re-infer final checkpoints without optimization."""
import argparse
from pathlib import Path
import numpy as np
import torch
import run as r


def replay(d, folder, device='cpu', tolerance=1e-4, inference=True):
    ck = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
    ident = ck['identity']
    train, valid, test = r.b.split(d, ident['outer_fold'], ident['inner_fold'])
    y = d['y'][train]
    aux = d['aux'][train]
    for name, expected in [('lo', r.b.scale_from_train(y)[0]), ('span', r.b.scale_from_train(y)[1]),
                           ('aux_lo', r.b.scale_from_train(aux)[0]), ('aux_span', r.b.scale_from_train(aux)[1])]:
        assert torch.equal(ck[name].cpu(), expected), 'Target scaling must use training records only'
    groups = [set(ident[k]) for k in ['train_people', 'validation_people', 'outer_evaluation_people']]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert groups[0] == set(d['person_id'][train])
    assert groups[1] == set(d['person_id'][valid])
    assert groups[2] == set(d['person_id'][test])
    assert ident['manifest_sha256']==r.b.sha(r.ROOT/'manifest.json')
    done=r.b.load(folder/'fit_complete.json')
    assert done['checkpoint_sha256']==r.b.sha(folder/'latest.pt') and done['curve_sha256']==r.b.sha(folder/'curve.json')
    assert done['identity']==ident and done['epochs']==len(ck['history'])==ident['epochs']
    assert r.b.load(folder/'curve.json')==ck['history']
    max_metric_error = 0.0
    max_coefficient_prediction_error = 0.0
    for row in ck['history']:
        with np.load(folder / ('epoch-%03d.npz' % row['epoch'])) as z:
            raw, coef = z['train_raw'], z['calibration']
            recalculated = r.b.calibrate(raw, y)
            max_coefficient_prediction_error = max(max_coefficient_prediction_error,
                    float(np.max(np.abs(r.b.corrected(raw, coef) - r.b.corrected(raw, recalculated)))))
            assert max_coefficient_prediction_error <= tolerance
            metric = r.b.person_mae(y, r.b.corrected(raw, coef), d['person_id'][train])
            max_metric_error = max(max_metric_error, abs(metric - row['train_person_mae']))
            if valid.any():
                metric = r.b.person_mae(d['y'][valid], r.b.corrected(z['validation_raw'], coef), d['person_id'][valid])
                max_metric_error = max(max_metric_error, abs(metric - row['validation_person_mae']))
    assert max_metric_error < 1e-9
    if not inference:
        return dict(passed=True,checkpoint_sha256=r.b.sha(folder/'latest.pt'),epochs=len(ck['history']),
                    max_metric_error=max_metric_error,max_refitted_calibration_prediction_error=max_coefficient_prediction_error)
    model = r.GdiCNN(r.config()['dropout']).double().to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    targets = d['x'][valid if valid.any() else test]
    values = []
    with torch.no_grad():
        for start in range(0, len(targets.values), 128):
            x = r.channels(torch.from_numpy(targets.values[start:start+128])).reshape(-1, 12, 124).double().to(device)
            values.append((model(x)[:,0] * ck['span'].double().to(device) + ck['lo'].double().to(device)).reshape(-1,2).cpu().numpy())
    actual = targets.aggregate(np.concatenate(values))
    if valid.any():
        with np.load(folder / ('epoch-%03d.npz' % len(ck['history']))) as z:
            expected, coef = z['validation_raw'], z['calibration']
    else:
        with np.load(folder.parent / 'predictions.npz') as z:
            expected, coef = z['predictions_raw'], z['calibration']
    raw_error = float(np.max(np.abs(actual - expected)))
    calibrated_error = float(np.max(np.abs(r.b.corrected(actual, coef) - r.b.corrected(expected, coef))))
    assert raw_error <= tolerance and calibrated_error <= tolerance, (raw_error, calibrated_error)
    return dict(passed=True, checkpoint_sha256=r.b.sha(folder/'latest.pt'), epochs=len(ck['history']),
                tolerance=tolerance, device=device, precision='float64', max_raw_error=raw_error,
                max_corrected_error=calibrated_error, max_metric_error=max_metric_error,
                max_refitted_calibration_prediction_error=max_coefficient_prediction_error)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--all',action='store_true')
    args=p.parse_args()
    r.b.setup(20261102,args.device)
    d=r.data()
    if args.all:
        paths=sorted(args.folder.glob('*/*/*/inner*/latest.pt'))+sorted(args.folder.glob('*/*/refit/latest.pt'))
        assert len(paths)==90
        audits={str(path.parent.relative_to(args.folder)):replay(d,path.parent,args.device,inference=False) for path in paths}
        reinference={}
        for path in sorted(args.folder.glob('*/*/refit/latest.pt')):
            reinference[str(path.parent.relative_to(args.folder))]=replay(d,path.parent,args.device)
        assert len(reinference)==10
        for variant in ['gdi_only','knee_auxiliary']:
            for outer in range(5):
                dest=args.folder/variant/('fold%d'%outer)
                selection=r.b.load(dest/'selection.json')
                weights=[0.0] if variant=='gdi_only' else r.config()['auxiliary_weights']
                candidates=[]
                for weight in weights:
                    histories=[r.b.load(dest/r.tag(weight)/('inner%d'%inner)/'curve.json') for inner in range(5) if inner!=outer]
                    epoch,curve=r.b.select_epoch(histories)
                    candidates.append(dict(weight=weight,epoch=epoch,score=curve[epoch-1],inner_curve=curve))
                assert selection['chosen']==r.select(candidates) and selection['candidates']==candidates
        result=dict(passed=True,audited_fits=90,audited_epochs=sum(v['epochs'] for v in audits.values()),
                    final_model_reinferences=10,audits=audits,reinference=reinference,
                    inner_weight_and_epoch_selection_recomputed=True,real_optimizer_steps=0)
    else:
        result=replay(d,args.folder,args.device)
    r.b.save(args.output,result)
    print(r.b.json.dumps(result),flush=True)


if __name__=='__main__':main()
