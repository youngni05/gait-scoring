from pathlib import Path
import json,urllib.request,hashlib,time,concurrent.futures,shutil
ROOT=Path('D:/Projects/D_P/studies/20261103-reference188-pilot')
DEST=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot/raw')
DEST.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(DEST).free>12_000_000_000
OLD=Path('D:/Projects/D_P/studies/20261102-sideview-reference-audit')
def download(article):
 meta=json.loads((OLD/f'figshare-{article}.json').read_text(encoding='utf-8-sig'));entry=meta['files'][0]
 fid=entry['id'];total=entry['size'];p=DEST/entry['name'];part=p.with_suffix(p.suffix+'.part')
 if not p.exists():
  for attempt in range(5):
   start=part.stat().st_size if part.exists() else 0
   if start==total:break
   req=urllib.request.Request(f'https://ndownloader.figshare.com/files/{fid}?download={time.time_ns()}',headers={'Range':f'bytes={start}-{total-1}'})
   try:
    with urllib.request.urlopen(req,timeout=60) as r:
     if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{total-1}/{total}':raise ValueError('Unexpected range')
     with part.open('ab') as out:
      last=start
      while b:=r.read(8*1024*1024):
       out.write(b);start+=len(b)
       if start-last>=256*1024*1024:print(fid,round(start/total*100,1),'%',flush=True);last=start
    break
   except Exception as e:
    print(fid,'retry',type(e).__name__,str(e),flush=True)
    if attempt==4:raise
  assert part.stat().st_size==total
  part.rename(p)
 md5=hashlib.md5();sha=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(16*1024*1024):md5.update(b);sha.update(b)
 expected=entry.get('supplied_md5') or entry.get('computed_md5')
 assert md5.hexdigest()==expected,(md5.hexdigest(),expected)
 row={'article_id':article,'file_id':fid,'path':str(p),'bytes':p.stat().st_size,'md5':md5.hexdigest(),'sha256':sha.hexdigest(),'provider_md5_match':True}
 print('VERIFIED',row,flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:rows=list(ex.map(download,[24192510,24192507]))
(ROOT/'raw-download-verification.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
