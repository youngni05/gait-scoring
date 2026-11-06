"""Acquire baseline bilateral clocks needed for phase coordination; no sensor reconstruction."""
from fetch import *
from concurrent.futures import ThreadPoolExecutor,as_completed
import time
name='gait2care';d=json.loads((S/(name+'-record.json')).read_text(encoding='utf-8'));f=next(x for x in d['files'] if x['key'].endswith('.zip'))
l=json.loads((S/(name+'-zip-index.json')).read_text());members=[x for x in l if x['name'].startswith('_IMU/Week0/') and x['size']]
starts=set()
for x in members:
 for k in range(x['offset']//4194304,(x['offset']+x['compressed']+256)//4194304+1):starts.add(k*4194304)
folder=B/name/'range-cache'
def block(start):
 p=folder/(str(start)+'.bin');end=min(start+4194304,f['size'])
 if p.exists():assert p.stat().st_size==end-start;return
 for attempt in range(3):
  try:
   req=urllib.request.Request(f['links']['self'],headers={'Range':f'bytes={start}-{end-1}'})
   with urllib.request.urlopen(req,timeout=90) as r:
    assert r.status==206;b=r.read()
   assert len(b)==end-start;p.write_bytes(b);return
  except Exception:
   if attempt==2:raise
   time.sleep(2)
# The gait-only fetch writes the block that also contains the first IMU; wait for it to complete externally before running this script.
with ThreadPoolExecutor(max_workers=6) as pool:
 for i,task in enumerate(as_completed([pool.submit(block,k) for k in sorted(starts)])):
  task.result()
  if i%20==0:print('baseline IMU blocks',i+1,'/',len(starts),flush=True)
z=zipfile.ZipFile(Remote(f['links']['self'],f['size'],folder));manifest=[]
for i,m in enumerate(members):
 x=z.getinfo(m['name']);p=B/name/'extracted'/x.filename;p.parent.mkdir(parents=True,exist_ok=True)
 b=z.read(x);p.write_bytes(b);manifest.append(dict(file=str(p),member=x.filename,sha256=hashlib.sha256(b).hexdigest(),crc32=x.CRC,crc_verified=True))
 if i%30==0:print('extracted baseline IMUs',i+1,flush=True)
save(S/'baseline-imu-manifest.json',manifest)
print('BASELINE_IMU_DONE',len(manifest),flush=True)
