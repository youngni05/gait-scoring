"""Hash and package without caches, checkpoints, or the upstream Git checkout."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
HERE=Path(__file__).resolve().parent;B=HERE/'bundle'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def included(p):return p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='manifest.json'
def main():
    shutil.copyfile(HERE.parent.parent/'cnn-frozen-residual692/repeats/capacity.py',B/'capacity.py')
    source=json.loads((B/'sources.json').read_text())
    for name,digest in source['upstream_files'].items():
        assert sha(B/'sources'/name)==digest
        if name.startswith('model/'):assert sha(B/name)==digest
    files=sorted(p for p in B.rglob('*') if included(p))
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),feature_version=(3,8))
    manifest={p.relative_to(B).as_posix():sha(p) for p in files}
    (B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    archive=HERE/'gdi-stt692-multiwindow.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[B/'manifest.json']:z.write(p,'gdi-stt692-multiwindow-20261031/'+p.relative_to(B).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read('gdi-stt692-multiwindow-20261031/'+name)).hexdigest()==digest
    result=dict(passed=True,files=len(manifest)+1,bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(B/'manifest.json'),
        python38_syntax_passed=True,zip_entries_hash_checked=True,official_model_unmodified=True,server_submitted=False)
    (HERE/'package_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
