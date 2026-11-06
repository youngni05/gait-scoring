"""Verify all reduced-input predictions, saved models, splits and comparisons."""
import sys, shutil, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import study
import numpy as np
import joblib

def main():
    d=study.load();inputs=np.load(HERE/'inputs.npz');worst=0.;n=0
    for name in ['major3d','motion3d','bend3d']:
        x=inputs[name]
        assert x.shape[0]==2953 and np.isfinite(x).all()
        for f in range(6):
            tr=np.flatnonzero((d['fold']!=f)&(d['fold']!=(f+1)%6));va=np.flatnonzero(d['fold']==(f+1)%6)
            outer=np.flatnonzero(d['fold']!=f);test=np.flatnonzero(d['fold']==f)
            for s in study.SEEDS:
                src=HERE/'received/fits'/name/('fold%d_seed%d.npz'%(f,s));z=np.load(src)
                for k,expected in [('indices',test),('train',outer),('inner_train',tr),('validation',va)]:assert np.array_equal(z[k],expected)
                leaf=int(z['selected_leaf']);choices=z['inner_choices'];assert leaf==choices[np.argmin(choices[:,0]),1]
                m=joblib.load(src.with_suffix('.joblib'));m.n_jobs=2
                assert m.random_state==s and m.n_estimators==200 and m.min_samples_leaf==leaf
                error=float(np.max(abs(m.predict(x[test])-z['prediction'])));worst=max(worst,error);assert error<=1e-12
                dst=HERE/'fits'/name/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
                # Preserve earlier local checkpoints; final canonical predictions come from the retrieved bundle.
                if dst.exists():assert np.array_equal(np.load(dst)['prediction'],z['prediction'])
                else:shutil.copyfile(src,dst)
                n+=1
    study.summarize()
    local=json.loads((HERE/'comparison.json').read_text());remote=json.loads((HERE/'received/comparison.json').read_text())
    for a,b in zip(local['seed_metrics'],remote['seed_metrics']):
        assert (a['variant'],a['seed'],a['group'])==(b['variant'],b['seed'],b['group'])
        for k in ['MAE','bias','large_error']:assert abs(a[k]-b[k])<1e-12
    result=dict(passed=True,models_replayed=n,max_replay_error=worst,people=110,walks=2953,seeds=study.SEEDS,folds=6,comparison_sha256=hashlib.sha256((HERE/'comparison.json').read_bytes()).hexdigest(),limited_to_derived_3d=True)
    study.write(HERE/'verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
