from acquire import HERE
import json,urllib.request,struct,zlib,hashlib,time,concurrent.futures
idx={x['name']:x for x in json.loads((HERE/'old-archive-index.json').read_text())}
matches=json.loads((HERE/'overlap-file-matches.json').read_text());chosen={}
for x in matches:
    if x['old'].endswith('/walk/walk.mot'):chosen.setdefault(x['old'].split('/')[1],x)
def get(pair):
    old_id,x=pair;i=idx[x['old']];dst=HERE/'overlap_raw'/f'{old_id}.mot';dst.parent.mkdir(exist_ok=True)
    if dst.exists():b=dst.read_bytes()
    else:
        start=i['offset'];end=start+i['compressed']+1023
        for attempt in range(6):
            try:
                req=urllib.request.Request('https://zenodo.org/records/13788592/files/opencap_data.zip',headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=60) as r:
                    assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-');raw=r.read()
                nl,el=struct.unpack_from('<HH',raw,26);b=zlib.decompress(raw[30+nl+el:30+nl+el+i['compressed']],-15);break
            except Exception:
                if attempt==5:raise
                time.sleep(20)
        dst.write_bytes(b);time.sleep(1)
    assert len(b)==i['size'] and zlib.crc32(b)==i['crc']
    return {'old_id':old_id,'old_file':x['old'],'new_file':x['new'][0],'sha256':hashlib.sha256(b).hexdigest()}
receipts=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for r in pool.map(get,chosen.items()):
        receipts.append(r);(HERE/'overlap-sha256.json').write_text(json.dumps(receipts,indent=2))
        if len(receipts)%20==0:print('OVERLAP',len(receipts),'/',len(chosen),flush=True)
print('OVERLAP_FETCH_COMPLETE',len(receipts),flush=True)
