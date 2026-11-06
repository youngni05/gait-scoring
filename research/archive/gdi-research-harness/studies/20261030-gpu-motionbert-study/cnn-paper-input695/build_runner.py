"""Create an isolated runner from the verified baseline; never edit previous bundle."""
from pathlib import Path
import ast,json,shutil
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'cnn-followup695/bundle'
B=HERE/'bundle'


def replace_function(s,name,code):
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=s.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+code.strip()+'\n'+''.join(lines[node.end_lineno:])


def main():
    s=(OLD/'run.py').read_text()
    s=s.replace('695-person matched-input CNN baseline','Official-feature matched-window CNN, retained 692-person cohort')
    s=s.replace('3 epochs on outer-0/inner-1 training people','3 epochs on outer-0/inner-1 training people (retained subset)')
    s=replace_function(s,'validate_data', '''
def validate_data(d):
    n=len(d['y'])
    require(d['x'].shape==(n,2,124,12) and d['x'].dtype==np.float32, 'Wrong side/time/feature layout')
    require(d['y'].shape==(n,2) and np.isfinite(d['x']).all() and np.isfinite(d['y']).all(), 'Invalid numbers')
    require(n==1061 and len(set(d['person_id']))==692 and len(set(d['video_id']))==n, 'Wrong retained cohort')
    for key in ('person_id','video_id','fold','cohort','original_row_index'):
        require(d[key].shape==(n,), 'Wrong metadata shape')
    require(len(np.unique(d['original_row_index']))==n, 'Duplicate source rows')
    require(set(d['fold'])==set(range(5)), 'Missing fold')
    for f in range(5):
        tr,_,te=split(d,f)
        require(len(set(d['person_id'][te]))==[138,138,138,139,139][f], 'Wrong fold membership')
        for j in range(5):
            if j!=f:split(d,f,j)
''')
    s=s.replace("c['stage'] == 'baseline_no_hyperparameter_search'", "c['stage'] == 'paper_feature_matched_window_no_search'")
    s=s.replace("c['target_scaling'] == 'training_only_per_side_minmax'", "c['target_scaling'] == 'training_only_pooled_sides_minmax'")
    s=s.replace("'earliest_minimum_of_four_inner_fold_person_MAE_by_epoch'", "'earliest_minimum_inner_person_weighted_calibrated_MAE'")
    s=s.replace("c['input'] == 'front124_h36m17_xy_confidence'", "c['input'] == 'matched124_official12_separate_sides'")
    s=replace_function(s,'scale_from_train','''
def scale_from_train(y):
    return torch.tensor(float(y.min()),dtype=torch.float32), torch.tensor(max(float(y.max()-y.min()),1e-6),dtype=torch.float32)
''')
    s=s.replace('(model(x) - target).square().mean()', '(model(x)[:,0] - target).square().mean()')
    s=replace_function(s,'predict','''
def predict(model,x,lo,span,device,batch=128):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):
            chunk=x[start:start+batch]
            p=model(chunk.reshape(-1,12,124).to(device))[:,0]*span.to(device)+lo.to(device)
            result.append(p.reshape(-1,2).cpu().numpy())
    return np.concatenate(result)


def calibrate(train_predictions,train_labels):
    # One shared slope/intercept, pooling left and right as in the source OLS code.
    design=np.column_stack([train_predictions.reshape(-1).astype(np.float64),np.ones(train_predictions.size)])
    coef=np.linalg.lstsq(design,train_labels.reshape(-1).astype(np.float64),rcond=None)[0]
    require(np.isfinite(coef).all(),'Invalid training-only calibration')
    return coef


def corrected(raw,coef):
    return raw.astype(np.float64)*coef[0]+coef[1]
''')
    s=s.replace("yt = (torch.from_numpy(y.astype(np.float32)) - lo) / span", "yt = ((torch.from_numpy(y.astype(np.float32)) - lo) / span).reshape(-1)\n    train_examples=xt.reshape(-1,12,124)")
    s=s.replace(".permutation(len(x))", ".permutation(len(yt))")
    s=s.replace("xt[idx].to(device), yt[idx].to(device)", "train_examples[idx].to(device), yt[idx].to(device)")
    s=s.replace("float(totals[0] / len(x))", "float(totals[0] / len(yt))").replace("float(totals[1] / len(x))", "float(totals[1] / len(yt))")
    s=s.replace("train_pred = predict(model, xt, lo, span, device)", "train_raw = predict(model, xt, lo, span, device)\n        correction=calibrate(train_raw,y)\n        train_pred=corrected(train_raw,correction)")
    s=s.replace("'train_person_mae': person_mae(y, train_pred, people)}", "'train_person_mae': person_mae(y, train_pred, people),\n               'train_raw_person_mae':person_mae(y,train_raw,people), 'calibration':correction.tolist()}")
    s=s.replace("row['validation_person_mae'] = person_mae(valid_y, predict(model, vx, lo, span, device), valid_people)", "val_raw=predict(model,vx,lo,span,device)\n            row['validation_person_mae']=person_mae(valid_y,corrected(val_raw,correction),valid_people)\n            row['validation_raw_person_mae']=person_mae(valid_y,val_raw,valid_people)\n            row['validation_people']=len(np.unique(valid_people))")
    s=s.replace("math.ceil(len(x) / c['batch_size'])", "math.ceil(len(yt) / c['batch_size'])")
    s=s.replace("means = curves.mean(axis=0)", "weights=np.array([h[0]['validation_people'] for h in histories],dtype=float)\n    means = np.average(curves,axis=0,weights=weights)")
    s=s.replace("# Each inner fold contains exactly 139 people, so equal fold weights equal person weights.", "# Subset fold sizes differ: weight by validation people, not by record count.")
    s=s.replace("pred = predict(model, channels(torch.from_numpy(d['x'][te])), lo, span, device)", "train_raw=predict(model,channels(torch.from_numpy(d['x'][tr])),lo,span,device)\n    coef=calibrate(train_raw,d['y'][tr])\n    raw=predict(model,channels(torch.from_numpy(d['x'][te])),lo,span,device)\n    pred=corrected(raw,coef)")
    s=s.replace("predictions=pred[:, None, :], y=d['y'][te],", "predictions=pred[:, None, :], predictions_raw=raw[:,None,:], calibration=coef, y=d['y'][te],")
    s=s.replace("'selected_epoch': epoch, 'torch':", "'calibration':coef.tolist(), 'raw_evaluation_person_mae':person_mae(d['y'][te],raw,d['person_id'][te]),\n                                'selected_epoch': epoch, 'torch':")
    # Output sizes and grouping reflect retained people, never claim all original 695.
    s=s.replace("np.full((1064, 2), np.nan)","np.full((len(d['y']), 2), np.nan)")
    s=s.replace("np.zeros(1064, dtype=int)","np.zeros(len(d['y']), dtype=int)")
    s=s.replace("('all695', np.ones(1064, dtype=bool))", "('retained_all', np.ones(len(d['y']), dtype=bool))")
    s=s.replace("('old175', d['cohort'] == 0)","('retained_old174', d['cohort'] == 0)").replace("('new520', d['cohort'] == 1)","('retained_new518', d['cohort'] == 1)")
    s=s.replace("range(1064)","range(len(d['y']))")
    s=s.replace("'passed': True, 'people': 695, 'records': 1064", "'passed': True, 'people': len(set(d['person_id'])), 'records': len(d['y'])")
    s=s.replace("'cnn_input_shape': [1064, 51, 124]", "'cnn_input_shape': [len(d['y'])*2, 12, 124]")
    s=s.replace("'train_people': 417, 'validation_people': 139", "'train_people': len(set(d['person_id'][tr])), 'validation_people':len(set(d['person_id'][va]))")
    s=s.replace("mean_seconds * c['max_epochs'] * (20 + 5 * 556 / 417)","mean_seconds*c['max_epochs']*sum(math.ceil(2*int(split(d,f,j)[0].sum())/c['batch_size']) for f in range(5) for j in ([None]+[k for k in range(5) if k!=f]))/math.ceil(2*int(tr.sum())/c['batch_size'])")
    s=s.replace("torch.rand(c['batch_size'], 124, 17, 3)", "torch.rand(c['batch_size']//2,2,124,12)")
    s=s.replace("x = channels(torch.rand(c['batch_size']//2,2,124,12)).to(device)","x = channels(torch.rand(c['batch_size']//2,2,124,12)).reshape(-1,12,124).to(device)")
    s=s.replace("y = torch.rand(c['batch_size'], 2).to(device)","y = torch.rand(c['batch_size']).to(device)")
    # Add raw-prediction OOF output as a diagnostic, without choosing raw/corrected on test errors.
    s=s.replace("seen = np.zeros(len(d['y']), dtype=int)","seen = np.zeros(len(d['y']), dtype=int)\n    raw_all=np.full_like(pred,np.nan)")
    s=s.replace("pred[idx] = z['predictions'][:, 0]", "pred[idx] = z['predictions'][:, 0]\n            raw_all[idx]=z['predictions_raw'][:,0]\n            np.testing.assert_allclose(pred[idx],corrected(raw_all[idx],z['calibration']),rtol=0,atol=1e-10)")
    s=s.replace("out / 'oof_predictions.npz', predictions=pred[:, None, :], y=d['y'],", "out / 'oof_predictions.npz', predictions=pred[:, None, :], predictions_raw=raw_all[:,None,:], original_row_index=d['original_row_index'], y=d['y'],")
    s=s.replace("'method': c['name'], 'metrics': rows, 'reference_metrics': reference,", "'method': c['name'], 'metrics': rows, 'reference_metrics': reference,\n               'raw_diagnostic_metrics':metrics(d['y'],raw_all,d['person_id']),\n               'primary_output':'training_only_OLS_corrected; raw is diagnostic',")
    (B/'run.py').write_text(s,encoding='utf-8')
    m=(OLD/'model.py').read_text()
    m=m.replace('51 if i == 0','12 if i == 0').replace('(51, 124)','(12, 124)').replace('B,51,124; 51 is channels','B,12,124; 12 is channels')
    m=m.replace('51 channels (17 joints x x/y/confidence), two meaningful outputs (left/right GDI).','12 channels per side; output 0 predicts that side; output 1 has zero loss weight, as in source.')
    m=replace_function(m,'channels','''
def channels(x):
    if x.ndim!=4 or tuple(x.shape[1:])!=(2,124,12):
        raise ValueError('Expected record, left/right, time124, features12')
    return x.transpose(2,3).contiguous()
''')
    (B/'model.py').write_text(m,encoding='utf-8')
    c=json.loads((OLD/'protocol.json').read_text())
    c.update(name='cnn_official12_matched124_calibrated_v1',stage='paper_feature_matched_window_no_search',
             target_scaling='training_only_pooled_sides_minmax',input='matched124_official12_separate_sides',
             selection='earliest_minimum_inner_person_weighted_calibrated_MAE',
             loss='equal_side_example_scaled_GDI_MSE_plus_kernel_L2',
             calibration='one_slope_and_intercept_fit_on_current_training_record_sides_only',
             primary_prediction='calibrated',note='Pinned public preprocessing and side-wise CNN; current person folds and person-MAE selection retained. Not full-paper reproduction.')
    (B/'protocol.json').write_text(json.dumps(c,indent=2)+'\n')
    for name in ('launch.sh','submit.slurm'):
        text=(OLD/name).read_text().replace('gdi-cnn695','gdi-cnn-paper').replace('CNN_PILOT_FINISHED','PAPER_CNN_PILOT_FINISHED').replace('CNN_TRAINING_AND_EXPORT_FINISHED','PAPER_CNN_TRAINING_AND_EXPORT_FINISHED')
        (B/name).write_bytes(text.encode())
    for p in B.glob('*.py'):ast.parse(p.read_text(),feature_version=(3,8))
    print('Runner prepared; single matched window only. No actual training.')


if __name__=='__main__':main()
