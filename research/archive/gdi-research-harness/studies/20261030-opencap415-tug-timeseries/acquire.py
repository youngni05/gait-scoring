from pathlib import Path
import io,zipfile,urllib.request,json,time,hashlib,sys
HERE=Path(__file__).resolve().parent
URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip'
class Remote(io.RawIOBase):
    def __init__(self,url,size): self.url=url;self.size=size;self.pos=0;self.start=0;self.buf=b''
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,n,w=0):
        self.pos=n if w==0 else self.pos+n if w==1 else self.size+n
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        if not (self.start<=self.pos and self.pos+n<=self.start+len(self.buf)):
            end=min(self.size-1,self.pos+max(n,262144)-1)
            for attempt in range(5):
                try:
                    req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{end}'})
                    with urllib.request.urlopen(req,timeout=60) as r:
                        assert r.status==206
                        self.buf=r.read();self.start=self.pos
                    break
                except Exception:
                    if attempt==4:raise
                    time.sleep(15)
            time.sleep(.5)
        out=self.buf[self.pos-self.start:self.pos-self.start+n];self.pos+=len(out);return out
def main():
    z=zipfile.ZipFile(Remote(URL,6584902803));names=z.namelist()
    info=[{'name':i.filename,'size':i.file_size,'compressed':i.compress_size,'crc':i.CRC,'offset':i.header_offset} for i in z.infolist()]
    (HERE/'archive-index.json').write_text(json.dumps(info,indent=2))
    selected=[n for n in names if n.endswith('.mot') and any(s in n.lower() for s in ['walk','gait'])]
    print('ARCHIVE',len(names),'WALK_MOT',len(selected),'SAMPLES',selected[:5],flush=True)
    (HERE/'selected-files.json').write_text(json.dumps(selected,indent=2))
    if '--index' in sys.argv:return
    receipts=[]
    for k,n in enumerate(selected):
        dst=HERE/'raw'/n;dst.parent.mkdir(parents=True,exist_ok=True)
        import zlib
        if dst.exists() and zlib.crc32(dst.read_bytes())==z.getinfo(n).CRC:b=dst.read_bytes()
        else:b=z.read(n);dst.write_bytes(b)
        assert zlib.crc32(b)==z.getinfo(n).CRC
        receipts.append({'name':n,'sha256':hashlib.sha256(b).hexdigest(),'crc_verified':True})
        (HERE/'acquisition.json').write_text(json.dumps(receipts,indent=2))
        if k%25==0:print('DOWNLOADED',k+1,'/',len(selected),flush=True)
    print('ACQUISITION_COMPLETE',len(receipts),flush=True)
if __name__=='__main__':main()
