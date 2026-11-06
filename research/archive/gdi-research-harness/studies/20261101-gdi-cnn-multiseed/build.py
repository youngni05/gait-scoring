"""Freeze the approved two-initialization experiment without changing baseline files."""
from pathlib import Path
import hashlib,json,shutil
H=Path(__file__).resolve().parent
BASE=H.parent/'20261030-gpu-motionbert-study/cnn-multiwindow695/bundle'
SEEDS=(20261101,20261102)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
def main():
    original=json.loads((BASE/'manifest.json').read_text())
    for n in ['run.py','model.py','windows.py','protocol.json','data.npz']:
        assert sha(BASE/n)==original[n],n
    source=(BASE/'run.py').read_text()
    source=replace(source,"with np.load(ROOT / 'data.npz', allow_pickle=False) as z:","path=Path(os.environ.get('GDI_DATA_PATH', str(ROOT.parent/'data.npz')))\n    require(sha(path)==load(ROOT/'protocol.json')['input_data_sha256'],'Input data changed')\n    with np.load(path, allow_pickle=False) as z:")
    source=replace(source,"    model = GdiCNN(c['dropout']).to(device)\n    opt = make_optimizer(model, c)","    model = GdiCNN(c['dropout']).to(device)\n    # Preserve the original post-initialization RNG state for training dropout.\n    if identity.get('initialization_seed',identity['fit_seed']) != identity['fit_seed']:\n        training_rng=rng_state()\n        setup(identity['initialization_seed'],device)\n        model=GdiCNN(c['dropout']).to(device)\n        restore_rng(training_rng)\n    opt = make_optimizer(model, c)")
    source=replace(source,"            'outer_fold': outer, 'inner_fold': inner, 'epochs': epochs, 'fit_seed': seed,","            'outer_fold': outer, 'inner_fold': inner, 'epochs': epochs, 'fit_seed': seed,\n            'initialization_seed':c['initialization_seed']+outer*100+(inner if inner is not None else 50),")
    source=replace(source,"        row['seconds'] = time.perf_counter() - started","        rawdir=folder/'raw_predictions';rawdir.mkdir(exist_ok=True)\n        rawfile=rawdir/('epoch%04d.npz'%(epoch+1))\n        np.savez_compressed(rawfile,train_raw=train_raw,validation_raw=val_raw if vx is not None else np.empty((0,2)))\n        row.update(raw_predictions_sha256=sha(rawfile),sample_sha256=hashlib.sha256(order.tobytes()).hexdigest(),sample_count=len(order))\n        row['seconds'] = time.perf_counter() - started")
    for seed in SEEDS:
        b=H/('seed%d'%seed);b.mkdir(exist_ok=True)
        (b/'run.py').write_text(source,encoding='utf-8',newline='\n')
        for n in ['model.py','windows.py','reference_metrics.json']:
            shutil.copyfile(BASE/n,b/n)
        shutil.copyfile(H/'support.py',b/'support.py')
        c=json.loads((BASE/'protocol.json').read_text())
        c.update(initialization_seed=seed,input_data_sha256=sha(BASE/'data.npz'))
        save(b/'protocol.json',c)
        save(b/'provenance.json',dict(original_manifest_sha256=sha(BASE/'manifest.json'),original_files={n:sha(BASE/n) for n in ['run.py','model.py','windows.py','protocol.json','data.npz']},changed_training_condition='model initialization only; original minibatch order and post-initialization dropout RNG retained'))
        files=['run.py','model.py','windows.py','reference_metrics.json','support.py','protocol.json','provenance.json']
        save(b/'manifest.json',{n:sha(b/n) for n in files})
    save(H/'protocol.json',dict(approved_initialization_seeds=list(SEEDS),baseline_seed=20261028,people=692,records=1061,windows=11261,baseline_mae=6.515816569178735,primary='For each seed average refit and four inner CNNs after member training-only OLS, exclude only training raw std<0.01 as in baseline; average three CNN seeds equally; retain baseline bagged STT and bilateral difference and existing fold-wise weights.',outer_selection=False,target=[6.1,6.3],deadline_kst='2026-11-01T23:59:59+09:00',expected_fits=50,expected_outer_folds=10,seed_selection='none',protected_final_data=False))
    print(json.dumps(dict(bundles=[str(H/('seed%d'%s)) for s in SEEDS],input_bytes=(BASE/'data.npz').stat().st_size)))
if __name__=='__main__':main()
