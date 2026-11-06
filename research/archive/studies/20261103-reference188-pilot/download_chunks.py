from pathlib import Path
import json,urllib.request,hashlib,time,concurrent.futures,shutil,threading,os
ROOT=Path('D:/Projects/D_P/studies/20261103-reference188-pilot')
DEST=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot/raw')
OLD=Path('D:/Projects/D_P/studies/20261102-sideview-reference-audit')
CHUNK=16*1024*1024
entries=[json.loads((OLD/f'figshare-{a}.json').read_text(encoding='utf-8-sig'))['files'][0] for a in [24192510,24192507]]
lock=threading.Lock();state={};completed=0
for e in entries:
 fid=e['id'];p=DEST/e['name'];part=p.with_suffix('.mat.part');manifest=DEST/(str(fid)+'-chunks.json')
 done=set(json.loads(manifest.read_text()) if manifest.exists() else [])
 if not manifest.exists():
  with part.open('wb') as f:f.truncate(e['size'])
 state[fid]={'entry':e,'path':p,'part':part,'manifest':manifest,'done':done}
jobs=[(e['id'],i) for e in entries for i in range((e['size']+CHUNK-1)//CHUNK) if i not in state[e['id']]['done']]
def fetch(job):
 global completed
 fid,idx=job;s=state[fid];total=s['entry']['size'];start=idx*CHUNK;stop=min(total,start+CHUNK)-1
 for attempt in range(5):
  try:
   req=urllib.request.Request(f'https://ndownloader.figshare.com/files/{fid}?part={idx}&t={time.time_ns()}',headers={'Range':f'bytes={start}-{stop}'})
   with urllib.request.urlopen(req,timeout=45) as r:
    assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{stop}/{total}'
    b=r.read(stop-start+2)
   assert len(b)==stop-start+1
   with lock:
    with s['part'].open('r+b') as f:f.seek(start);f.write(b)
    s['done'].add(idx);s['manifest'].write_text(json.dumps(sorted(s['done'])))
    completed+=1
    if completed%16==0:print('download',completed,'/',len(jobs),'chunks',round(completed*CHUNK/1e9,2),'GB',flush=True)
   return
  except Exception as e:
   if attempt==4:raise
   time.sleep(1)
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:list(ex.map(fetch,jobs))
rows=[]
for fid,s in state.items():
 md5=hashlib.md5();sha=hashlib.sha256()
 with s['part'].open('rb') as f:
  while b:=f.read(16*1024*1024):md5.update(b);sha.update(b)
 expected=s['entry'].get('supplied_md5') or s['entry'].get('computed_md5')
 assert md5.hexdigest()==expected,(fid,md5.hexdigest(),expected)
 s['part'].rename(s['path']);rows.append({'file_id':fid,'path':str(s['path']),'bytes':s['entry']['size'],'md5':md5.hexdigest(),'sha256':sha.hexdigest(),'provider_md5_match':True})
 print('VERIFIED',fid,flush=True)
(ROOT/'raw-download-verification.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
