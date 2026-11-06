"""Seal code and data provenance, then build a task-specific transfer archive."""
import ast,hashlib,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE/'bundle'
BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={}
for p in sorted(B.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='manifest.json':continue
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),feature_version=(3,8))
    manifest[p.relative_to(B).as_posix()]=sha(p)
(B/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=BIG/'ljmu100-mixed-20261104.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in B.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(B).as_posix())
    z.write(BIG/'data.npz','data/data.npz')
    for p in (BIG/'baseline').rglob('*.pt'):z.write(p,'data/'+p.relative_to(BIG).as_posix())
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
        'code_manifest_sha256':sha(B/'manifest.json'),'python38_syntax_passed':True}
(HERE/'package.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
