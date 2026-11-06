from pathlib import Path
import json,urllib.request,struct,zlib,hashlib,time,concurrent.futures
from acquire import HERE,URL
index=json.loads((HERE/'archive-index.json').read_text())
selected=[x for x in index if x['name'].endswith('_task-10mwt.mot')]
(HERE/'selected-files.json').write_text(json.dumps(selected,indent=2))
print('SELECTED',len(selected),'COMPRESSED_MB',sum(x['compressed'] for x in selected)/1e6,flush=True)
def get(x):
    dst=HERE/'raw'/x['name'];dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and zlib.crc32(dst.read_bytes())==x['crc']:b=dst.read_bytes()
    else:
        start=x['offset'];end=start+x['compressed']+1023
        for attempt in range(6):
            try:
                req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=60) as r:
                    assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-')
                    raw=r.read()
                assert raw[:4]==b'PK\x03\x04'
                nl,el=struct.unpack_from('<HH',raw,26);offset=30+nl+el
                b=zlib.decompress(raw[offset:offset+x['compressed']],-15)
                assert len(b)==x['size'] and zlib.crc32(b)==x['crc']
                dst.write_bytes(b);break
            except Exception:
                if attempt==5:raise
                time.sleep(15)
        time.sleep(1)
    return {'name':x['name'],'sha256':hashlib.sha256(b).hexdigest(),'crc_verified':True}
receipt=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for r in pool.map(get,selected):
        receipt.append(r);(HERE/'acquisition.json').write_text(json.dumps(receipt,indent=2))
        if len(receipt)%20==0:print('DOWNLOADED',len(receipt),'/',len(selected),flush=True)
print('ACQUISITION_COMPLETE',len(receipt),flush=True)
