"""Public source pilot: official checksums, workbook and ZIP member inventory only."""
from pathlib import Path
import requests,json,hashlib,io,zipfile,time
HERE=Path(__file__).resolve().parent
DEST=Path('D:/gait-research-data/D_P/planar-tug-improvement-20261104/gstride')
def get(url,headers=None):
    for attempt in range(3):
        try:
            r=requests.get(url,headers=headers,timeout=60);r.raise_for_status();return r
        except Exception:
            if attempt==2:raise
            time.sleep(2)
class Remote(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.total=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,w=0):self.pos=n if w==0 else self.pos+n if w==1 else self.size+n;return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n<=0:return b''
        assert n<8*1024*1024 and self.total+n<16*1024*1024
        end=self.pos+n-1;r=get(self.url,{'Range':f'bytes={self.pos}-{end}'})
        assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {self.pos}-{end}/') and len(r.content)==n
        self.pos+=n;self.total+=n;return r.content
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    meta=get('https://zenodo.org/api/records/17052815').json();(DEST/'zenodo-record.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    files=meta['files'];out=[]
    for f in files:
        if f['key'].endswith('.xlsx'):
            b=get(f['links']['self']).content
            assert hashlib.md5(b).hexdigest()==f['checksum'].split(':')[1]
            p=DEST/f['key'];p.write_bytes(b)
            out.append({'file':f['key'],'bytes':len(b),'md5':hashlib.md5(b).hexdigest(),'sha256':hashlib.sha256(b).hexdigest(),'official_checksum_verified':True})
    f=next(x for x in files if x['key'].endswith('.zip'));remote=Remote(f['links']['self'],f['size']);z=zipfile.ZipFile(remote)
    index=[{'name':x.filename,'size':x.file_size,'compressed':x.compress_size,'offset':x.header_offset,'crc':x.CRC} for x in z.infolist()]
    (DEST/'zip-inventory.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    audit={'source':'https://zenodo.org/records/17052815','license':meta['metadata'].get('license'),'files':out,'archive_members':len(index),
      'zip_bytes':f['size'],'range_bytes':remote.total,'whole_archive_downloaded':False,'example_members':[x['name'] for x in index[:25]]}
    (HERE/'gstride-acquisition.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
