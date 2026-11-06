"""Create a bounded, hash-checked public-frame-only GPU submission bundle."""
from pathlib import Path
import hashlib,json,shutil,tarfile
HERE=Path(__file__).resolve().parent
ROOT=Path('D:/Projects/D_P')
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    rows=json.loads((OUT/'input-manifest.json').read_text())
    assert len(rows)==12 and sum(r['frames'] for r in rows)==1488
    package=OUT/'package';package.mkdir(exist_ok=True)
    for name in ('pose_worker.py','pose.slurm'):shutil.copyfile(HERE/name,package/name)
    shutil.copyfile(ROOT/'gdi-research-harness/studies/20261031-cnn-high-gdi/research/pose_torch.py',package/'pose_torch.py')
    shutil.copyfile(OUT/'input-manifest.json',package/'input-manifest.json')
    files={p.name:sha(p) for p in package.iterdir() if p.is_file() and p.name!='package-hashes.json'}
    for r in rows:
        assert sha(OUT/'inputs'/r['file'])==r['sha256']
        files['inputs/'+r['file']]=r['sha256']
    network=ROOT/'gdi-research-harness/studies/20261031-cnn-high-gdi/research/pose_probe'
    hashes={'files':files,'network':{n:sha(network/n) for n in ('network.json','network_weights.npz','native_arrays.npz')}}
    (package/'package-hashes.json').write_text(json.dumps(hashes,indent=2))
    archive=OUT/'pose-submission.tar'
    with tarfile.open(archive,'w') as tar:
        for name in sorted(files):tar.add(OUT/name if name.startswith('inputs/') else package/name,arcname=name,recursive=False)
        tar.add(package/'package-hashes.json',arcname='package-hashes.json',recursive=False)
    record={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'people':12,'frames':1488,
            'remote':'/abr/RESEARCH_USER/gdi-tug-transfer-20261031','purpose':'Official BODY25 forward inference on selected public Toronto frames; no clinical labels uploaded; existing public network reused read-only'}
    (OUT/'package-receipt.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
