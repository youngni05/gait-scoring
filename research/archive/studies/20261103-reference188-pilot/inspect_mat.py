import io,sys,json,time,urllib.request,hashlib
from pathlib import Path
ROOT=Path('D:/Projects/D_P/studies/20261103-reference188-pilot')
sys.path.insert(0,str(ROOT/'deps'))
import h5py
CACHE=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot/ranges')
class Remote(io.RawIOBase):
 def __init__(self,fid,size):
  self.fid=fid;self.size=size;self.pos=0;self.block=262144;self.calls=0
  self.cache=CACHE/str(fid);self.cache.mkdir(parents=True,exist_ok=True)
  with urllib.request.urlopen(urllib.request.Request('https://ndownloader.figshare.com/files/'+str(fid),headers={'Range':'bytes=0-0'}),timeout=60) as r:self.url=r.url
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def readinto(self,b):
  data=self.read(len(b));b[:len(data)]=data;return len(data)
 def read(self,n=-1):
  end=self.size if n<0 else min(self.pos+n,self.size);out=[]
  while self.pos<end:
   idx=self.pos//self.block;start=idx*self.block;stop=min(start+self.block,self.size)-1;p=self.cache/(str(idx)+'.bin')
   if not p.exists():
    if self.calls>=1200:raise RuntimeError('300 MiB network budget reached')
    for attempt in range(3):
     try:
      req=urllib.request.Request('https://ndownloader.figshare.com/files/'+str(self.fid)+'?range='+str(start)+'&time='+str(time.time_ns()),headers={'Range':f'bytes={start}-{stop}'})
      with urllib.request.urlopen(req,timeout=60) as r:
       if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{stop}/{self.size}':raise RuntimeError('Invalid range response')
       data=r.read(stop-start+2)
      if len(data)!=stop-start+1:raise RuntimeError('Truncated range')
      p.write_bytes(data);self.calls+=1;break
     except Exception:
      if attempt==2:raise
      time.sleep(1)
   data=p.read_bytes();count=min(end-self.pos,len(data)-(self.pos-start));out.append(data[self.pos-start:self.pos-start+count]);self.pos+=count
  return b''.join(out)
def describe(obj):
 return {'type':type(obj).__name__,'shape':list(obj.shape) if isinstance(obj,h5py.Dataset) else None,'dtype':str(obj.dtype) if isinstance(obj,h5py.Dataset) else None,'keys':list(obj.keys()) if isinstance(obj,h5py.Group) else None}
if __name__=='__main__':
 result={}
 for name,fid,size in [('healthy',42450096,6244593847),('stroke',42450087,2906058754)]:
  remote=Remote(fid,size)
  with h5py.File(remote,'r') as f:
   data=f['Sub'];row={'root':list(f.keys()),'data':describe(data)}
   for key in data:
    obj=data[key];row[key]=describe(obj)
    if isinstance(obj,h5py.Group):row[key]['children']={k:describe(obj[k]) for k in obj}
   result[name]=row;print(name,json.dumps(row),flush=True)
  print('network blocks',remote.calls,flush=True)
 (ROOT/'mat-schema.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
