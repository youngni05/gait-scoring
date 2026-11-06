"""Retrieve only twelve prespecified bottom-camera members, validating ZIP CRC.

The first study reuses the Toronto participant set; it does not use TUG values
to choose cameras or intervals. Original OAW06 already exists and is reused.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, io, json, struct, urllib.request, zipfile, zlib

ROOT=Path('D:/Projects/D_P')
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
URL='https://ndownloader.figshare.com/files/35454242'
SIZE=2351609541
IDS=[1,2,3,5,6,7,9,10,11,12,13,14]

def response(first,last):
    req=urllib.request.Request(URL,headers={'Range':'bytes=%d-%d'%(first,last)})
    r=urllib.request.urlopen(req,timeout=180)
    assert r.status==206 and r.headers['Content-Range'].startswith('bytes %d-'%first)
    return r

class Remote(io.RawIOBase):
    def __init__(self):self.pos=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else SIZE+n
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else SIZE-self.pos,SIZE-self.pos)
        if n==0:return b''
        assert n<2000000
        with response(self.pos,self.pos+n-1) as r:b=r.read()
        assert len(b)==n
        self.pos+=n;return b

def inspect(p):
    h=hashlib.sha256();crc=0;size=0
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b);crc=zlib.crc32(b,crc);size+=len(b)
    return h.hexdigest(),crc&0xffffffff,size

def get(info):
    person=int(Path(info.filename).name[3:5]);dest=OUT/'videos'/Path(info.filename).name
    if person==6:dest=ROOT/'outputs/gait-public-access-20261015/Toronto/OAW06-bottom.mp4'
    if not dest.exists():
        assert person!=6
        with response(info.header_offset,info.header_offset+29) as r:head=r.read()
        assert head[:4]==b'PK\x03\x04'
        namelen,extra=struct.unpack_from('<HH',head,26)
        start=info.header_offset+30+namelen+extra
        temp=dest.with_suffix('.part')
        decoder=zlib.decompressobj(-15) if info.compress_type==8 else None
        assert info.compress_type in (0,8)
        received=0
        with response(start,start+info.compress_size-1) as r,temp.open('wb') as stream:
            while True:
                b=r.read(1024*1024)
                if not b:break
                received+=len(b);stream.write(decoder.decompress(b) if decoder else b)
            if decoder:stream.write(decoder.flush());assert decoder.eof
        assert received==info.compress_size
        digest,crc,size=inspect(temp)
        assert crc==info.CRC and size==info.file_size
        temp.replace(dest)
    digest,crc,size=inspect(dest)
    assert crc==info.CRC and size==info.file_size
    row={'person':str(person),'camera':'bottom','member':info.filename,'path':str(dest.resolve()),'sha256':digest,'crc32':crc,'bytes':size}
    (OUT/'acquisition'/('%02d.json'%person)).write_text(json.dumps(row,indent=2))
    print('verified',person,size,flush=True)
    return row

def main():
    assert OUT.resolve().drive.upper()=='D:'
    for name in ('videos','acquisition'):(OUT/name).mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(Remote()) as z:
        infos=[z.getinfo('Videos/OAW%02d-bottom.mp4'%p) for p in IDS]
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(get,infos))
    result={'source_archive':URL,'source_archive_bytes':SIZE,'verification':'member CRC32, uncompressed size, recorded SHA256; full archive MD5 not claimed',
            'people':len(rows),'scope':'same 12 participants as existing Toronto TUG experiment; one prespecified bottom camera','videos':rows}
    (OUT/'acquisition.json').write_text(json.dumps(result,indent=2))
    print('ACQUISITION_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
