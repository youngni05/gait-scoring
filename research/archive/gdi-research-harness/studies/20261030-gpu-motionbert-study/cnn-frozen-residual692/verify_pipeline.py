"""Synthetic end-to-end five-fold audit, isolated from real input and outputs."""
from pathlib import Path
import sys,tempfile,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'bundle'))
import run


def main():
    out=HERE/'checks/pipeline'
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(99)
    n=10
    data=dict(fold=np.arange(n)%5,person_id=np.array(['synthetic-%d'%i for i in range(n)]),
        video_id=np.array(['video-%d'%i for i in range(n)]),cohort=np.zeros(n,int),original_row_index=np.arange(n),
        y=rng.normal(73,1,(n,2)))
    features=rng.normal(size=(n,124,512)).astype(np.float32)
    baseline=np.full((n,2),70.,dtype=float)
    c=dict(name='synthetic_only_pipeline',seed=20261028,learning_rate=.001,weight_decay=.0001,batch_size=4,max_epochs=2)
    old_root=run.ROOT
    old_base=run.baseline
    def fake_base(d,f,j,device,rows):
        run.split(d,f,j)
        return baseline[rows].copy()
    try:
        with tempfile.TemporaryDirectory(dir=str(out),prefix='fixture-') as tmp:
            root=Path(tmp)
            run.ROOT=root
            run.baseline=fake_base
            for name in ('run.py','residual.py','base_model.py','protocol.json','sources.json','manifest.json'):
                (root/name).write_text('{}',encoding='utf-8')
            np.save(root/'single_oof.npy',baseline,allow_pickle=False)
            for f in range(5):
                for j in [k for k in range(5) if k!=f]+[None]:
                    path=run.baseline_path(f,j)
                    path.mkdir(parents=True,exist_ok=True)
                    (path/'latest.pt').write_bytes(b'synthetic checkpoint identity only')
                run.train_fold(data,features,c,f,root/'results','cpu')
            run.summarize(data,c,root/'results')
            summary=run.load(root/'results/summary.json')
            assert summary['fits']==25 and summary['people']==n and summary['records']==n
            with np.load(root/'results/oof_predictions.npz',allow_pickle=False) as z:
                expected=float(np.mean(np.abs(z['predictions']-data['y'])))
                assert abs(expected-summary['mae'])<1e-12
            target=root/'results/fold0/predictions.npz'
            target.write_bytes(target.read_bytes()+b'corrupt')
            rejected=False
            try:run.summarize(data,c,root/'results')
            except ValueError:rejected=True
            assert rejected
            result=dict(passed=True,synthetic_only=True,real_GDI_optimizer_steps=0,synthetic_fits=25,
                synthetic_optimizer_steps=summary['optimizer_steps'],five_fold_export_and_independent_metric=True,
                corrupt_output_rejected=True)
            run.save(out/'verification.json',result)
            print(json.dumps(result))
    finally:
        run.ROOT=old_root
        run.baseline=old_base


if __name__=='__main__':main()
