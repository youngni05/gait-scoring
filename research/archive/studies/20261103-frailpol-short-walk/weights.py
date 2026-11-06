from pathlib import Path
import requests,hashlib,json,time
H=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');B=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103')
old=json.loads(Path('D:/Projects/D_P/studies/20261103-sppb-tabular-foundation/weights-manifest.json').read_text(encoding='utf8'));out=old.copy()
for repo,name in [('jingang/TabICL','tabicl-classifier-v2-20260212.ckpt'),('Prior-Labs/TabPFN-v2-clf','tabpfn-v2-classifier-v2_default.ckpt')]:
    meta=requests.get('https://huggingface.co/api/models/'+repo,params={'blobs':'true'},timeout=60);meta.raise_for_status();meta=meta.json();rev=meta['sha'];blob=next(s for s in meta['siblings'] if s['rfilename']==name)
    path=B/'weights'/name;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true&request_time={int(time.time())}'
        with requests.get(url,stream=True,timeout=60) as r:
            r.raise_for_status()
            with path.with_suffix('.part').open('wb') as f:
                for chunk in r.iter_content(4*1024*1024):f.write(chunk)
        path.with_suffix('.part').replace(path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==blob['lfs']['sha256']
    out.append(dict(repo=repo,revision=rev,name=name,path=str(path),sha256=sha,bytes=path.stat().st_size))
    print(name,path.stat().st_size,sha,flush=True)
(H/'weights-manifest.json').write_text(json.dumps(out,indent=2),encoding='utf8')
