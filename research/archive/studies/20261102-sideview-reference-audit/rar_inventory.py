"""Read public RAR headers by byte range, without downloading all video data."""
import io, json, sys, urllib.request, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'deps'))
import rarfile

URL = 'https://data.mendeley.com/public-files/datasets/44pfnysy89/files/10e4e5b6-753a-4326-bbed-d41e563a61a9/file_downloaded'
SIZE = 4091995786
class Remote(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.start = 0
        self.cache = b''
        self.requests = 0
        self.total = 0
        self.url = (HERE/'koa-range-url.txt').read_text(encoding='utf-8-sig').strip()
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else SIZE + offset
        return self.pos
    def read(self, n=-1):
        if n < 0: n = SIZE - self.pos
        if self.pos >= SIZE: return b''
        if not (self.start <= self.pos and self.pos+n <= self.start+len(self.cache)):
            end = min(SIZE-1, self.pos + max(n,8192)-1)
            req = urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{end}'})
            with urllib.request.urlopen(req,timeout=45) as r:
                assert r.status == 206, r.status
                assert r.headers['Content-Range'].startswith(f'bytes {self.pos}-')
                self.url = r.url
                self.cache = r.read()
            self.start = self.pos
            self.requests += 1
            self.total += len(self.cache)
            if self.requests % 20 == 0: print('header requests',self.requests,flush=True)
        out=self.cache[self.pos-self.start:self.pos-self.start+n]
        self.pos += len(out)
        return out

remote=Remote()
archive=rarfile.RarFile(remote)
records=[]
for f in archive.infolist():
    records.append({k:getattr(f,k,None) for k in ['filename','file_size','compress_size','header_offset','data_offset','CRC','compress_type','flags']})
result={'source':URL,'archive_size':SIZE,'range_requests':remote.requests,'downloaded_bytes':remote.total,'entries':records}
(HERE/'koa-rar-inventory.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'count':len(records),'bytes_read':remote.total,'first':records[:3],'last':records[-3:]},indent=2),flush=True)
