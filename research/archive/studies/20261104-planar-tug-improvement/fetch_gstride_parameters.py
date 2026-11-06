"""Fetch compact processed CSV members, not the 1.3GB raw signal archive."""
from pathlib import Path
import json,struct,zlib,hashlib,sys
from fetch_gstride import get,DEST,HERE
def main():
    index=json.loads((DEST/'zip-inventory.json').read_text());meta=json.loads((DEST/'zenodo-record.json').read_text())
    url=next(x['links']['self'] for x in meta['files'] if x['key'].endswith('.zip'))
    segment='--segments' in sys.argv
    prefix='_GAIT/_SEGMENTATION/' if segment else '_GAIT/_PARAMETERS/'
    selected=[x for x in index if x['name'].startswith(prefix) and x['name'].endswith('.csv')]
    # Official cohort163; actualV3 archive inventory contains158 processed parameter files.
    assert len(selected)==158
    groups=[selected,[next(x for x in index if x['name']=='Readme.txt')]];receipts=[];network=0
    for group in groups:
        start=min(x['offset'] for x in group);end=max(x['offset']+x['compressed']+1023 for x in group)
        assert end-start<32*1024*1024
        r=get(url,{'Range':f'bytes={start}-{end}'})
        assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        buf=r.content;assert len(buf)==end-start+1;network+=len(buf)
        for x in group:
            pos=x['offset']-start;assert buf[pos:pos+4]==b'PK\x03\x04'
            method=struct.unpack_from('<H',buf,pos+8)[0];nl,el=struct.unpack_from('<HH',buf,pos+26)
            assert buf[pos+30:pos+30+nl].decode()==x['name']
            raw=buf[pos+30+nl+el:pos+30+nl+el+x['compressed']]
            b=zlib.decompress(raw,-15) if method==8 else raw
            assert len(b)==x['size'] and zlib.crc32(b)==x['crc']
            path=DEST/'processed'/x['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
            receipts.append({'member':x['name'],'sha256':hashlib.sha256(b).hexdigest(),'crc32':x['crc'],'bytes':len(b)})
    outname='gstride-segmentation-acquisition.json' if segment else 'gstride-parameters-acquisition.json'
    (HERE/outname).write_text(json.dumps({'advertised_people':163,'processed_member_files':158,'network_bytes':network,'members':receipts,'whole_archive_downloaded':False},indent=2),encoding='utf-8')
    print('ACQUIRED',len(receipts),'MEMBERS',network,'NETWORK_BYTES',flush=True)
    for x in selected[:1]:print((DEST/'processed'/x['name']).read_text()[:2500])
    print((DEST/'processed/Readme.txt').read_text()[:7000])
if __name__=='__main__':main()
