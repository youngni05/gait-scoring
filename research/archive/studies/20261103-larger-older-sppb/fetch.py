"""Fetch public workbook and ZIP gait members by HTTP Range; preserve CRC and SHA256."""
from pathlib import Path
import urllib.request,json,hashlib,io,zipfile,sys,collections
sys.stdout.reconfigure(encoding='utf-8')
S=Path(__file__).resolve().parent
B=Path('D:/gait-research-data/D_P/larger-older-sppb-20261103')
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
class Remote(io.RawIOBase):
 def __init__(self,u,n,folder):self.u=u;self.n=n;self.p=0;self.folder=folder;folder.mkdir(parents=True,exist_ok=True)
 def seekable(self):return True
 def tell(self):return self.p
 def seek(self,p,w=0):self.p=p if w==0 else self.p+p if w==1 else self.n+p;return self.p
 def read(self,n=-1):
  n=min(n if n>=0 else self.n-self.p,self.n-self.p)
  if n<=0:return b''
  out=[]
  while n:
   start=self.p//4194304*4194304;end=min(start+4194304,self.n);f=self.folder/(str(start)+'.bin')
   if not f.exists():
    req=urllib.request.Request(self.u,headers={'Range':f'bytes={start}-{end-1}'})
    with urllib.request.urlopen(req,timeout=90) as r:
     if r.status!=206:raise RuntimeError('Range not supported '+str(r.status))
     b=r.read()
    assert len(b)==end-start;f.write_bytes(b)
   b=f.read_bytes();k=min(n,end-self.p);out.append(b[self.p-start:self.p-start+k]);self.p+=k;n-=k
  return b''.join(out)
def main():
 for name,rid in [('gait2care',17017009),('gstride',17052815)]:
  rp=S/(name+'-record.json')
  if not rp.exists():save(rp,json.load(urllib.request.urlopen('https://zenodo.org/api/records/'+str(rid),timeout=40)))
  d=json.loads(rp.read_text(encoding='utf-8'));manifest=[]
  for f in d['files']:
   if f['key'].endswith('.xlsx'):
    p=B/f['key']
    if not p.exists():p.write_bytes(urllib.request.urlopen(f['links']['self'],timeout=60).read())
    assert hashlib.md5(p.read_bytes()).hexdigest()==f['checksum'].split(':')[1]
    manifest.append(dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source=f['links']['self'],md5_verified=True));continue
   z=zipfile.ZipFile(Remote(f['links']['self'],f['size'],B/name/'range-cache'))
   lst=[dict(name=x.filename,size=x.file_size,compressed=x.compress_size,crc=x.CRC,offset=x.header_offset) for x in z.infolist()]
   save(S/(name+'-zip-index.json'),lst)
   c=collections.Counter()
   for x in lst:c['/'.join(x['name'].split('/')[:-1])]+=x['compressed']
   print(name,'FOLDERS',dict(c),flush=True)
   selected=[x for x in z.infolist() if not x.is_dir() and (x.filename.lower().endswith(('.txt','.md','.pdf')) or '_GAIT/' in x.filename or 'gait_parameters' in x.filename.lower() or 'segmentation' in x.filename.lower())]
   print(name,'selected',len(selected),sum(x.file_size for x in selected),flush=True)
   for i,x in enumerate(selected):
    p=B/name/'extracted'/x.filename
    assert p.resolve().is_relative_to((B/name/'extracted').resolve())
    if p.exists():b=p.read_bytes()
    else:b=z.read(x);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    import zlib
    assert zlib.crc32(b)==x.CRC
    manifest.append(dict(file=str(p),member=x.filename,sha256=hashlib.sha256(b).hexdigest(),crc32=x.CRC,source=f['links']['self'],crc_verified=True))
    if i%50==0:print(name,'fetched',i,flush=True)
  save(S/(name+'-manifest.json'),manifest)
  print(name,'DONE',len(manifest),flush=True)
if __name__=='__main__':main()
