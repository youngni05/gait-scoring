from pathlib import Path
import json,urllib.request,hashlib,time,concurrent.futures,shutil
HERE=Path('D:/Projects/D_P/studies/20261103-frozen-transfer-and-deviation');HERE.mkdir(parents=True,exist_ok=True)
BIG=Path('D:/gait-research-data/D_P/frozen-transfer-and-deviation-20261103');BIG.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(BIG).free>6_000_000_000
META=Path('D:/Projects/D_P/studies/20261102-sideview-reference-audit')
def one(article):
 d=json.loads((META/f'figshare-{article}.json').read_text(encoding='utf-8-sig'));e=d['files'][0];p=BIG/e['name'];part=p.with_suffix('.zip.part')
 if not p.exists():
  start=part.stat().st_size if part.exists() else 0
  if start<e['size']:
   req=urllib.request.Request(e['download_url']+'?run='+str(time.time_ns()),headers={'Range':f'bytes={start}-{e["size"]-1}'})
   with urllib.request.urlopen(req,timeout=60) as r:
    assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{e["size"]-1}/{e["size"]}'
    with part.open('ab') as out:
     while b:=r.read(8*1024*1024):out.write(b)
  assert part.stat().st_size==e['size'];part.rename(p)
 md5=hashlib.md5();sha=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):md5.update(b);sha.update(b)
 assert md5.hexdigest()==e['computed_md5']
 row={'article':article,'path':str(p),'bytes':p.stat().st_size,'sha256':sha.hexdigest(),'md5':md5.hexdigest(),'license':d['license'],'verified':True}
 (HERE/f'download-{article}.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:rows=list(ex.map(one,[24192480,24192483]))
(HERE/'downloads.json').write_text(json.dumps(rows,indent=2))
