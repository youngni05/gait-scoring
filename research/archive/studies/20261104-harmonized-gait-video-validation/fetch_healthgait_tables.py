"""Read ZIP directory stream and extract only three clinical/reference tables."""
from pathlib import Path
import requests,json,struct,zlib,hashlib
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104/healthgait')
def getpart(f,start,end):
    r=requests.get(f['links']['self'],headers={'Range':f'bytes={start}-{end}'},timeout=60);r.raise_for_status()
    assert r.status_code==206 and len(r.content)==end-start+1
    assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
    return r.content
def main():
    meta=json.loads((BIG/'record.json').read_text());f=next(x for x in meta['files'] if x['key']=='Health_Gait.zip')
    tail=(BIG/'central-tail.bin').read_bytes();p=tail.rfind(b'PK\x05\x06');h=struct.unpack_from('<4s4H2LH',tail,p)
    size,offset=h[5],h[6];assert h[1]==h[2]==25 and size<256*1024**2
    found=[];buf=b'';network=0;seen=0
    with requests.get(f['links']['self'],headers={'Range':f'bytes={offset}-{offset+size-1}'},stream=True,timeout=60) as r:
      r.raise_for_status();assert r.status_code==206
      for chunk in r.iter_content(1024*1024):
        network+=len(chunk);buf+=chunk;pos=0
        while len(buf)-pos>=46:
          a=struct.unpack_from('<4s6H3L5H2L',buf,pos);assert a[0]==b'PK\x01\x02'
          n=46+a[10]+a[11]+a[12]
          if len(buf)-pos<n:break
          name=buf[pos+46:pos+46+a[10]].decode('utf8');seen+=1
          if name.lower().endswith('.csv'):
            found.append({'name':name,'disk':a[13],'offset':a[16],'compressed':a[8],'bytes':a[9],'crc32':a[7],'method':a[4]})
            print('CSV_ENTRY',found[-1],flush=True)
          pos+=n
        buf=buf[pos:]
        if len(found)>=3:break
        if network%(20*1024**2)==0:print('DIRECTORY_MB',network//1024**2,'ENTRIES',seen,flush=True)
    receipts=[]
    for x in found:
        assert x['bytes']<10*1024**2 and x['disk']<=25
        part=next(y for y in meta['files'] if y['key']==('Health_Gait.zip' if x['disk']==25 else f'Health_Gait.z{x["disk"]+1:02d}'))
        block=getpart(part,x['offset'],x['offset']+x['compressed']+2047)
        assert block[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',block,26)
        assert block[30:30+nl].decode()==x['name']
        raw=block[30+nl+el:30+nl+el+x['compressed']];data=zlib.decompress(raw,-15) if x['method']==8 else raw
        assert len(data)==x['bytes'] and zlib.crc32(data)==x['crc32']
        path=BIG/'tables'/Path(x['name']).name;path.parent.mkdir(exist_ok=True);path.write_bytes(data)
        receipts.append({**x,'sha256':hashlib.sha256(data).hexdigest(),'local':str(path)})
    (HERE/'healthgait-tables-acquisition.json').write_text(json.dumps({'directory_network_bytes':network,'entries_scanned':seen,'tables':receipts,'full_dataset_downloaded':False},indent=2),encoding='utf8')
    print('TABLES_VERIFIED',len(receipts),flush=True)
if __name__=='__main__':main()
