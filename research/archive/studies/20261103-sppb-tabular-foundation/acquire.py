from pathlib import Path
import json, hashlib, urllib.request, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path('D:/Projects/D_P/studies/20261103-sppb-tabular-foundation')
BIG=Path('D:/gait-research-data/D_P/sppb-tabular-foundation-20261103')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    records=[]
    for repo, filename, licensefile in [('jingang/TabICL','tabicl-regressor-v2-20260212.ckpt','LICENSE'),('Prior-Labs/TabPFN-v2-reg','tabpfn-v2-regressor-v2_default.ckpt','LICENSE.txt')]:
        with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true',timeout=60) as r: meta=json.load(r)
        revision=meta['sha']; dest=BIG/'weights'/repo.split('/')[-1];dest.mkdir(parents=True,exist_ok=True)
        (dest/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
        for name in [filename,licensefile,'README.md']:
            entry=next((s for s in meta['siblings'] if s['rfilename']==name),None)
            if entry is None:
                if name==licensefile and not meta.get('cardData',{}).get('license'): raise RuntimeError('License missing: '+repo)
                continue
            path=dest/name;url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
            if not path.exists(): urllib.request.urlretrieve(url,path)
            digest=sha(path);expected=entry.get('lfs',{}).get('sha256')
            if expected: assert digest==expected,(name,digest,expected)
            records.append(dict(repo=repo,revision=revision,name=name,url=url,path=str(path),bytes=path.stat().st_size,sha256=digest,expected_lfs_sha256=expected))
            print(name,path.stat().st_size,digest,flush=True)
    (HERE/'weights-manifest.json').write_text(json.dumps(records,indent=2),encoding='utf8')
if __name__=='__main__': main()
