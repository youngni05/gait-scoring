"""Seal a new, reviewable feature-only school-GPU bundle."""
import hashlib,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/child100-train-only-20261105')
BUNDLE=HERE/'bundle'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
manifest={str(p.relative_to(BUNDLE)).replace('\\','/'):sha(p) for p in sorted(BUNDLE.rglob('*'))
          if p.is_file() and p!=BUNDLE/'manifest.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
save(BUNDLE/'manifest.json',manifest)
target=BIG/'child100-train-only-20261105.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in list(manifest)+['manifest.json']:
        z.write(BUNDLE/name,'child100-train-only-20261105/'+name)
    z.write(BIG/'data.npz','child100-train-only-20261105/data/data.npz')
save(HERE/'bundle-seal.json',dict(zip_path=str(target),zip_bytes=target.stat().st_size,
    zip_sha256=sha(target),manifest_sha256=sha(BUNDLE/'manifest.json'),data_sha256=sha(BIG/'data.npz'),
    contents='Coordinates/features, assumed and measured labels, anonymous people IDs, ages, original model code and protocol. No RGB videos.'))
print((HERE/'bundle-seal.json').read_text())
