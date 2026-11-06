"""Read provider inventories and ZIP members without downloading entire archives."""
import io, json, hashlib, urllib.request, zipfile, time
from pathlib import Path
import concurrent.futures

HERE = Path(__file__).resolve().parent
BIG = Path('D:/gait-research-data/D_P/kuopio-movi-audit-20261105')

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def fetch(url, start=None, end=None):
    headers = {'User-Agent':'GaitResearch-source-audit/1.0','Accept-Encoding':'identity'}
    if start is not None: headers['Range'] = f'bytes={start}-{end}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),timeout=60) as r:
        if start is not None:
            assert r.status == 206, (r.status, dict(r.headers))
            assert r.headers['Content-Range'].startswith(f'bytes {start}-'), dict(r.headers)
        value=r.read()
    if start is not None: assert len(value)==end-start+1
    return value

class RemoteZip(io.RawIOBase):
    def __init__(self,url,size): self.url,self.size,self.pos,self.bytes=url,size,0,0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self, n=-1):
        if n<0: n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if not n:return b''
        assert n<400_000_000, n
        v=fetch(self.url,self.pos,self.pos+n-1);self.pos+=len(v);self.bytes+=len(v)
        return v

def inventory(file):
    name=file['key']; target=BIG/(name+'.members.json')
    if target.exists(): return json.loads(target.read_text())
    remote=RemoteZip(file['links']['self'],file['size'])
    with zipfile.ZipFile(remote) as z:
        members=[{'name':x.filename,'size':x.file_size,'compressed':x.compress_size,'crc32':x.CRC,
                  'offset':x.header_offset} for x in z.infolist()]
    result={'archive':name,'size':file['size'],'url':file['links']['self'],'transferred':remote.bytes,'members':members}
    dump(target,result)
    print(json.dumps({'archive':name,'members':len(members),'read_bytes':remote.bytes}),flush=True)
    return result

def main():
    BIG.mkdir(parents=True, exist_ok=True)
    raw=fetch('https://zenodo.org/api/records/10559504');(HERE/'kuopio-provider.json').write_bytes(raw)
    data=json.loads(raw)
    for f in data['files']:
        if not f['key'].endswith('.zip'):
            content=fetch(f['links']['self']);(HERE/f['key']).write_bytes(content)
            assert 'md5:'+hashlib.md5(content).hexdigest()==f['checksum']
    print('small provider files downloaded and checksum verified',flush=True)
    files=[f for f in data['files'] if f['key'].endswith('.zip')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(inventory,files))
    dump(HERE/'kuopio-inventory-summary.json',[{k:v for k,v in x.items() if k!='members'}|{'members':len(x['members'])} for x in results])

if __name__=='__main__':main()
