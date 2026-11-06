"""Assemble a new immutable bundle from already verified sources; no training."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
B=HERE/'bundle'
S=HERE.parent/'cnn-paper-input695'
F=HERE.parent/'cnn-motionbert-fusion692'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    assert read(S/'analysis/verification.json')['passed']
    assert read(F/'analysis/verification.json')['passed']
    manifest=read(S/'bundle/manifest.json')
    for n,h in manifest.items():assert sha(S/'bundle'/n)==h,n
    for src,dst in [(S/'bundle/data.npz',B/'data.npz'),(S/'bundle/model.py',B/'base_model.py'),
                    (F/'received/cache/features.npy',B/'features.npy')]:
        shutil.copyfile(src,dst)
    (B/'sources').mkdir(exist_ok=True)
    for name in ('LICENSE','provenance.json'):
        shutil.copyfile(S/'bundle/sources'/name,B/'sources'/name)
    shutil.copyfile(S/'bundle/run.py',B/'sources/base_run.py')
    shutil.copyfile(S/'bundle/manifest.json',B/'sources/base_manifest.json')
    shutil.copyfile(S/'bundle/protocol.json',B/'sources/base_protocol.json')
    with np.load(F/'bundle/motion_input.npz',allow_pickle=False) as z, np.load(B/'data.npz',allow_pickle=False) as d:
        for k in ('person_id','video_id','fold','original_row_index'):np.testing.assert_array_equal(z[k],d[k])
        np.save(B/'pooled_reference.npy',z['pooled_reference'],allow_pickle=False)
    with np.load(S/'analysis/cnn_oof.npz',allow_pickle=False) as z:
        np.save(B/'single_oof.npy',z['predictions'][:,0],allow_pickle=False)
    copied=[]
    for f in range(5):
        for name in ['inner%d'%j for j in range(5) if j!=f]+['refit']:
            src=S/'received/results'/('fold%d'%f)/name
            dest=B/'base'/('fold%d'%f)/name
            info=read(src/'fit_complete.json')
            assert sha(dest/'latest.pt')==info['checkpoint_sha256'],(f,name)
            shutil.copyfile(src/'fit_complete.json',dest/'fit_complete.json')
            copied.append(dict(fold=f,fit=name,sha256=info['checkpoint_sha256']))
    cache=read(F/'received/bundle/complete.json')
    assert sha(B/'features.npy')==cache['features_sha256']
    save(B/'sources.json',dict(base_bundle_sha256=sha(S/'bundle/manifest.json'),base_data_sha256=sha(B/'data.npz'),
        feature_cache_sha256=cache['features_sha256'],feature_cache_proof=cache,base_checkpoints=copied,
        frozen_cnn=True,frozen_motionbert=True,base_source='gdi-cnn-paper-input-20261030',
        feature_source='gdi-cnn-motionbert-fusion-20261030/cache/temporal/features.npy'))
    files=sorted(p for p in B.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts)
    save(B/'manifest.json',{p.relative_to(B).as_posix():sha(p) for p in files})
    target=HERE/'gdi-cnn-frozen-residual.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files+[B/'manifest.json']:z.write(p,'gdi-cnn-frozen-residual-20261030/'+p.relative_to(B).as_posix())
    save(HERE/'package_verification.json',dict(zip_sha256=sha(target),bytes=target.stat().st_size,
        manifest_sha256=sha(B/'manifest.json'),files=len(files)+1,baseline_checkpoints=25))
    print((HERE/'package_verification.json').read_text())


if __name__=='__main__':main()
