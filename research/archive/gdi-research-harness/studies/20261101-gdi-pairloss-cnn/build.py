"""Freeze two approved loss arms, preserving the original baseline and input."""
from pathlib import Path
import hashlib,json,shutil
H=Path(__file__).resolve().parent
BASE=H.parent/'20261030-gpu-motionbert-study/cnn-multiwindow695/bundle'
AUDIT=H.parent/'20261101-gdi-longitudinal-audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def change(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
def main():
    frozen=json.loads((BASE/'manifest.json').read_text())
    for name in ['run.py','model.py','windows.py','protocol.json','data.npz']:assert sha(BASE/name)==frozen[name]
    visits=json.loads((AUDIT/'development-visits.json').read_text());byvideo={v:r for r in visits for v in r['video_ids']}
    # Read only already-authorized development metadata; no reserved input.
    import numpy as np
    with np.load(BASE/'data.npz',allow_pickle=False) as z:
        metadata=[dict(person_id=str(p),video_id=str(v),age=byvideo[str(v)]['age_values'][0] if len(byvideo[str(v)]['age_values'])==1 else None)
                  for p,v in zip(z['person_id'],z['video_id'])]
    assert len(metadata)==1061
    source=(BASE/'run.py').read_text()
    source=change(source,"from windows import WindowSet, weighted_mse","from windows import WindowSet, weighted_mse\nfrom pair_ops import pool_for, paired_step")
    source=change(source,"with np.load(ROOT / 'data.npz', allow_pickle=False) as z:","path=Path(os.environ.get('GDI_DATA_PATH', str(ROOT.parent/'data.npz')))\n    require(sha(path)==load(ROOT/'protocol.json')['input_data_sha256'],'Input changed')\n    with np.load(path, allow_pickle=False) as z:")
    source=change(source,"    validate_data(d)\n    return d","    m=load(ROOT/'visit_metadata.json')\n    require([r['video_id'] for r in m]==d['video_id'].tolist() and [r['person_id'] for r in m]==d['person_id'].tolist(),'Visit ordering changed')\n    d['x'].ages=np.array([r['age'] if r['age'] is not None else np.nan for r in m])\n    validate_data(d)\n    return d")
    source=change(source,"    vx = valid_x","    normalized_labels=(torch.from_numpy(y.astype(np.float32))-lo)/span\n    context=(x,normalized_labels,pool_for(x,people)) if c['pair_weight'] else None\n    vx = valid_x")
    source=change(source,"        totals = np.zeros(2)","        totals = np.zeros(3)")
    source=change(source,"np.asarray(step(model, opt, train_examples[idx].to(device), yt[idx].to(device), c, example_weights[idx].to(device)))","np.asarray(paired_step(step,model,opt,train_examples[idx].to(device),yt[idx].to(device),c,example_weights[idx].to(device),context,identity['fit_seed']+epoch*100000+start,device))")
    source=change(source,"'scaled_mse': float(totals[1] / len(yt)),","'scaled_mse': float(totals[1] / len(yt)), 'scaled_pair_mse':float(totals[2]/len(yt)),")
    source=change(source,"        row['seconds'] = time.perf_counter() - started","        rawdir=folder/'raw_predictions';rawdir.mkdir(exist_ok=True)\n        rawfile=rawdir/('epoch%04d.npz'%(epoch+1))\n        np.savez_compressed(rawfile,train_raw=train_raw,validation_raw=val_raw if vx is not None else np.empty((0,2)))\n        row.update(raw_predictions_sha256=sha(rawfile),sample_sha256=hashlib.sha256(order.tobytes()).hexdigest(),sample_count=len(order))\n        row['seconds'] = time.perf_counter() - started")
    windows=(BASE/'windows.py').read_text()
    windows=change(windows,"starts=None,fallback=None):","starts=None,fallback=None,ages=None):")
    windows=change(windows,"        self.values=values","        self.ages=ages\n        self.values=values")
    windows=change(windows,"self.starts[keep],self.fallback[keep])","self.starts[keep],self.fallback[keep],None if self.ages is None else self.ages[indices])")
    support=(H.parent/'20261101-gdi-cnn-multiseed/support.py').read_text()
    first=support.index('def verify(');last=support.index('\ndef worker():',first)
    support=support[:first]+"def verify(out,device):\n    from verify_impl import run_checks\n    return run_checks(r,out,device)\n"+support[last:]
    support=change(support,"'seed':c['initialization_seed']","'seed':c['pair_weight']") if "'seed':c['initialization_seed']" in support else support.replace("seed=c['initialization_seed']","seed=c['pair_weight']")
    # Same two distinct GPUs, one arm per worker. Main starts only after local pilot review.
    support=support.replace('>3600:', '>21600:')
    for arm in (0,1):
        b=H/('seed%d'%arm);b.mkdir(exist_ok=True)
        (b/'run.py').write_text(source,encoding='utf-8',newline='\n');(b/'windows.py').write_text(windows,encoding='utf-8',newline='\n')
        (b/'support.py').write_text(support,encoding='utf-8',newline='\n')
        for name in ['model.py','reference_metrics.json']:shutil.copyfile(BASE/name,b/name)
        shutil.copyfile(BASE/'run.py',b/'baseline_run.py')
        for name in ['pair_ops.py','verify_impl.py']:shutil.copyfile(H/name,b/name)
        shutil.copyfile(BASE/'sources/LICENSE',b/'LICENSE')
        save(b/'visit_metadata.json',metadata)
        c=json.loads((BASE/'protocol.json').read_text());c.update(pair_weight=arm,input_data_sha256=sha(BASE/'data.npz'),wall_time_seconds=150000,
            pair_objective='uniform training person then pair; all-window mean; normalized GDI delta squared; eval-mode differentiable branch',
            baseline_optimizer_steps_preserved=True,comparison_compute_matched=False)
        save(b/'protocol.json',c)
        save(b/'provenance.json',dict(baseline_manifest_sha256=sha(BASE/'manifest.json'),source_files={n:sha(BASE/n) for n in ['run.py','model.py','windows.py','protocol.json','data.npz']},arm=arm))
        save(b/'manifest.json',{p.name:sha(p) for p in sorted(b.iterdir()) if p.is_file() and p.name!='manifest.json'})
    save(H/'protocol.json',dict(approved=True,owner_thread='01a19552-3eb5-7020-9424-1e27c9d7f936',automation_id='gdi',
         arms=[0,1],expected_fits=50,expected_outer_folds=10,people=692,records=1061,windows=11261,max_epochs=100,
         same_seed=20261028,pair_loss_weight=1,epoch_selection='unchanged: inner calibrated person visit MAE',
         scope='Existing CNN only; compare matched loss arms, not a new version of the old CNN/STT ensemble',
         primary_report=['same-person change MAE','change direction','left/right','first age-ordered visit MAE','all-visit MAE'],
         clinical_threshold=None,success_threshold=None,protected_final_opened=False,
         data_sha256=sha(BASE/'data.npz'),approval='User selected 기존 CNN으로 새 학습 비교 진행'))
    print(json.dumps({'built_arms':[0,1],'real_training_steps':0}))
if __name__=='__main__':main()
