from pathlib import Path
import ast,hashlib,json,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent; B=HERE/'bundle';BASE=HERE.parent/'cnn-paper-input695'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    with np.load(B/'data.npz',allow_pickle=False) as z, np.load(BASE/'bundle/data.npz',allow_pickle=False) as old:
        for key in ('y','person_id','video_id','fold','cohort','original_row_index'):
            assert np.array_equal(z[key],old[key]),key
        assert z['x'].shape==(11261,2,124,12)
    contract=json.loads((HERE/'input_audit.json').read_text())
    contract.update(record_count=1061,people=692,axes=['window','left_right','time124','feature12'],
        window_to_record='window_record',window_start='zero based, frame = window_start + 1',
        window_rule='range(0,500-124,31); required xy max missing <=25%; finite official features',
        canonical_fallback_records=4,source_single_bundle_sha256=sha(BASE/'bundle/manifest.json'),
        preprocessing='same pinned full-clip processing as single window; ambiguity kept missing',
        target_and_fold='record-level; identical y, IDs and folds to single-window bundle',
        loss='global-mean-one inverse number-of-windows per RECORD; not per-person balance',
        batch_normalization='unweighted window distribution; inverse weighting affects MSE only',
        inference='uniform window mean per record side, then train-record-only pooled-side OLS',
        not_compute_matched=True,protected_labels_opened=False)
    save(B/'input_contract.json',contract)
    for p in B.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    for p in list(B.glob('*.sh'))+list(B.glob('*.slurm')):p.write_bytes(p.read_text().replace('\r\n','\n').encode())
    files=[p for p in sorted(B.rglob('*')) if p.is_file() and p.name!='manifest.json' and not any(k in p.parts for k in ('__pycache__','checks','runs','logs'))]
    manifest={p.relative_to(B).as_posix():sha(p) for p in files};save(B/'manifest.json',manifest)
    archive=HERE/'gdi-cnn-multiwindow.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[B/'manifest.json']:z.write(p,'gdi-cnn-multiwindow-20261030/'+p.relative_to(B).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read('gdi-cnn-multiwindow-20261030/'+name)).hexdigest()==digest
    result={'passed':True,'zip_sha256':sha(archive),'zip_bytes':archive.stat().st_size,'bundle_sha256':sha(B/'manifest.json'),'files':len(manifest)+1}
    save(HERE/'package_verification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
