from pathlib import Path
import ast,hashlib,json,zipfile
HERE=Path(__file__).resolve().parent;B=HERE/'bundle'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    for p in B.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    for p in list(B.glob('*.sh'))+list(B.glob('*.slurm')):p.write_bytes(p.read_text().replace('\r\n','\n').encode())
    files=[p for p in sorted(B.rglob('*')) if p.is_file() and p.name!='manifest.json' and not any(k in p.parts for k in ('__pycache__','checks','runs','logs','cache'))]
    manifest={p.relative_to(B).as_posix():sha(p) for p in files};save(B/'manifest.json',manifest)
    archive=HERE/'gdi-cnn-motionbert-fusion.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[B/'manifest.json']:z.write(p,'gdi-cnn-motionbert-fusion-20261030/'+p.relative_to(B).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read('gdi-cnn-motionbert-fusion-20261030/'+name)).hexdigest()==digest
    result=dict(passed=True,zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,bundle_sha256=sha(B/'manifest.json'),files=len(manifest)+1,actual_gpu_execution=False)
    save(HERE/'package_verification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
