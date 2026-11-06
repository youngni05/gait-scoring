"""Build an isolated experiment from the audited CNN; never edit the source bundle."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
S=H.parent/'cnn-record-loss692/bundle'
def write(n,s): (H/n).write_text(s,encoding='utf-8')
def change(s,a,b):
    assert a in s,a[:100]
    return s.replace(a,b)
for n in ['model.py','windows.py','reference_metrics.json','input_contract.json']:
    write(n,(S/n).read_text(encoding='utf-8-sig'))
s=(S/'run.py').read_text(encoding='utf-8-sig')
s=change(s,'from record_loss import packed, recording_step','from balanced import sample_epoch, batches, sample_digest, objective, steps_per_epoch')
s=change(s,"np.load(ROOT / 'data.npz', allow_pickle=False)","np.load(os.environ.get('GDI_DATA_PATH', str(ROOT / 'data.npz')), allow_pickle=False)")
s=change(s,"    validate_data(d)\n    return d","    validate_data(d)\n    require(sha(os.environ.get('GDI_DATA_PATH', str(ROOT/'data.npz'))) == config()['data_sha256'], 'Wrong source data')\n    return d")
s=change(s,'record_loss_matched_batch_no_search','person_balanced_mse_bmc_fixed_sigma')
s=change(s,"'batch_records'","'batch_size'")
s=change(s,"['window_control','record_mean']","['mse','bmc']")
s=change(s,"'_packed_records'","'_person_balanced'")
s=change(s,"ROOT / 'record_loss.py'","ROOT / 'balanced.py'")
a=s.index('        order = np.random.default_rng',s.index('def fit('));b=s.index('        train_raw = predict',a)
s=s[:a]+'''        order = sample_epoch(x, people, identity['fit_seed'] + epoch)
        totals = np.zeros(2)
        for idx in batches(len(order), c['batch_size']):
            chosen=order[idx]; win=chosen[:,0]; side=chosen[:,1]
            inputs=torch.from_numpy(x.values[win,side]).transpose(1,2).contiguous().to(device)
            targets=yt[x.owner[win],side].to(device)
            model.train();opt.zero_grad(set_to_none=True)
            pred=model(inputs)[:,0]
            data_loss=objective(pred,targets,c['loss_variant'],c['sigma_gdi']/float(span))
            loss=(data_loss+c['kernel_l2']*model.penalty())*len(idx)/c['batch_size']
            require(bool(torch.isfinite(loss)), 'Nonfinite loss')
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Invalid gradients')
            opt.step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Invalid updated parameters')
            totals+=np.array([float(data_loss.detach()),float((pred-targets).square().mean().detach())])*len(idx)
''' +s[b:]
s=change(s,'totals[0] / len(yt)','totals[0] / len(order)')
s=change(s,'totals[1] / len(yt)','totals[1] / len(order)')
s=change(s,"        row['raw_predictions_sha256']=sha(rawfile)","        row['raw_predictions_sha256']=sha(rawfile)\n        row['sample_sha256']=sample_digest(order)\n        row['sample_count']=len(order)\n        row['samples_per_person']=len(order)//len(np.unique(people))")
s=change(s,"len(history) * math.ceil(len(yt) / c['batch_size'])","len(history) * steps_per_epoch(x, people, c['batch_size'])")
s=change(s,"math.ceil(len(d['x'][split(d,f,j)[0]])/c['batch_size'])","steps_per_epoch(d['x'][split(d,f,j)[0]], d['person_id'][split(d,f,j)[0]], c['batch_size'])")
s=change(s,"math.ceil(len(d['x'][tr])/c['batch_size'])","steps_per_epoch(d['x'][tr],d['person_id'][tr],c['batch_size'])")
# Retain historical record-pooled TRAIN-only OLS in BOTH arms: sampler/loss are the new factors.
s=change(s,"'record_balanced_loss':True","'person_balanced_sampling':True") if "'record_balanced_loss':True" in s else s
write('run.py',s)
a=(S/'audit.py').read_text()
a=change(a,"epochs*int(np.ceil(len(y)/c['batch_records']))","epochs*r.steps_per_epoch(x,ids,c['batch_size'])")
a=change(a,"        r.require(row['epoch']==k,'Nonsequential curve')","        r.require(row['epoch']==k,'Nonsequential curve')\n        order=r.sample_epoch(x,ids,ident['fit_seed']+k-1)\n        r.require(row['sample_sha256']==r.sample_digest(order) and row['sample_count']==len(order), 'Wrong person-balanced samples')\n        r.require(row['samples_per_person']==len(order)//len(np.unique(ids)), 'Wrong sample mass')")
write('audit.py',a)
c=json.loads((S/'protocol.json').read_text())
for k in ['batch_records','baseline_batch_size','historical_baseline_manifest_sha256']:c.pop(k,None)
c.update(name='cnn_person_balanced_comparison',stage='person_balanced_mse_bmc_fixed_sigma',batch_size=32,
    variants=['mse','bmc'],sigma_gdi=5.0,wall_time_seconds=82000,
    data_sha256='7c1912cd27c63a82ed0d0324f98c0c0fae9a608174d52bd7a502f05e9a8c7a1e',
    loss='person-balanced sampler; MSE vs official BMC with fixed sigma=5 GDI; identical L2',
    note='Exact equal count per person per epoch; uniform record then window and side. Same draws/initialization across arms. No outer-selected parameters.',
    workers=3,expected_fits=50,expected_outer_folds=10)
write('protocol.json',json.dumps(c,indent=2)+'\n')
print('Prepared isolated CNN adaptation')
