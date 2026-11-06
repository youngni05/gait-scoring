import hashlib,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE/'bundle';BIG=Path('D:/gait-research-data/D_P/matched-normal100-20261105')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((HERE/'preflight.json').read_text())['passed']
assert not (B/'manifest.json').exists(),'Do not reseal a submitted run'
manifest={str(p.relative_to(B)).replace('\\','/'):sha(p) for p in sorted(B.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
target=BIG/'matched-normal100-20261105.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in list(manifest)+['manifest.json']:z.write(B/name,'matched-normal100-20261105/'+name)
    for arm in ['adult','child']:z.write(BIG/arm/'data.npz','matched-normal100-20261105/data/'+arm+'/data.npz')
out=dict(zip_path=str(target),zip_bytes=target.stat().st_size,zip_sha256=sha(target),manifest_sha256=sha(B/'manifest.json'),contents='Stanford and matched normal motion features, assumed/measured labels, anonymous IDs, ages, weights and code. No RGB videos or private credentials.')
(HERE/'bundle-seal.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))
