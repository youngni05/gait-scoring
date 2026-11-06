import argparse, concurrent.futures, hashlib, io, json, re, struct, time, urllib.request, zipfile, zlib, sys, gzip
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path('D:/gait-research-data/D_P/ljmu-side-audit-20261104')
URL = 'https://opendata.ljmu.ac.uk/id/eprint/133/3/Gait%20Dataset.zip'
SIZE = 18700445600
sys.path.insert(0,str(HERE.parents[1]/'tmp/ljmu-audit-deps'))
from inflate64 import Inflater

def get(start, end):
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise ValueError('Unexpected range response')
                b = r.read()
            if len(b) != end-start+1: raise ValueError('Short range')
            return b
        except Exception:
            if attempt == 3: raise
            time.sleep(2)

def inventory():
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA/'central-directory.bin'
    if not p.exists():
        tail = get(SIZE-128, SIZE-1)
        pos = tail.index(b'PK\x06\x06')
        h = struct.unpack_from('<4sQHHIIQQQQ', tail, pos)
        cd_size, cd_start = h[-2:]
        p.write_bytes(get(cd_start, cd_start+cd_size-1))
    cd = p.read_bytes()
    fake = cd + struct.pack('<4sHHHHIIH',b'PK\x05\x06',0,0,65535,65535,len(cd),0,0)
    infos = zipfile.ZipFile(io.BytesIO(fake)).infolist()
    files = []
    for z in infos:
        if z.filename.lower().endswith(('.mp4','.xlsx','.txt')):
            files.append(dict(name=z.filename,size=z.file_size,compressed=z.compress_size,
                              offset=z.header_offset,crc=z.CRC,method=z.compress_type))
    (HERE/'inventory.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    side = [x for x in files if x['name'].lower().endswith('.mp4') and re.search(r'_90(?:_ALT)?\.mp4$',x['name'],re.I)]
    print(json.dumps({'entries':len(infos),'videos':sum(x['name'].lower().endswith('.mp4') for x in files),
                      'side_candidates':len(side),'side_bytes':sum(x['size'] for x in side),
                      'samples':[x['name'] for x in side[:6]],'metadata':[x['name'] for x in files if x['name'].endswith('.xlsx')]}),flush=True)

def fetch(e):
    dest = DATA/'files'/e['name']
    if dest.exists() and dest.stat().st_size == e['size']:
        b=dest.read_bytes()
        if zlib.crc32(b)&0xffffffff == e['crc']: return dict(name=e['name'],ok=True,cached=True)
    h = get(e['offset'],e['offset']+29)
    vals=struct.unpack('<4sHHHHHIIIHH',h)
    if vals[0]!=b'PK\x03\x04': raise ValueError('Bad local header')
    start=e['offset']+30+vals[-2]+vals[-1]
    c=get(start,start+e['compressed']-1)
    if e['method']==9: b=Inflater().inflate(c)
    elif e['method']==8: b=zlib.decompress(c,-15)
    elif e['method']==0: b=c
    else: raise ValueError('Unsupported compression')
    if len(b)!=e['size'] or zlib.crc32(b)&0xffffffff!=e['crc']: raise ValueError('Member integrity failure')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(b)
    return dict(name=e['name'],ok=True,sha256=hashlib.sha256(b).hexdigest())

def download(kind):
    es=json.loads((HERE/'inventory.json').read_text())
    if kind=='metadata': es=[e for e in es if e['name'].endswith('.xlsx')]
    elif kind=='side': es=[e for e in es if re.search(r'_90(?:_ALT)?\.mp4$',e['name'],re.I)]
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures={pool.submit(fetch,e):e for e in es}
        for f in concurrent.futures.as_completed(futures):
            e=futures[f]
            try: out.append(f.result())
            except Exception as ex: out.append(dict(name=e['name'],ok=False,error=str(ex)))
            if len(out)%25==0: print(f'{len(out)}/{len(es)} downloaded; failed={sum(not x["ok"] for x in out)}',flush=True)
    (HERE/f'download-{kind}.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'total':len(out),'ok':sum(x['ok'] for x in out)}),flush=True)

def poses():
    cd=(DATA/'central-directory.bin').read_bytes()
    fake=cd+struct.pack('<4sHHHHIIH',b'PK\x05\x06',0,0,65535,65535,len(cd),0,0)
    groups={}
    for e in zipfile.ZipFile(io.BytesIO(fake)).infolist():
        folder=e.filename.rsplit('/',1)[0]
        if e.filename.endswith('.json') and re.search(r'_90(?:_ALT)?_json$',folder,re.I):
            groups.setdefault(folder,[]).append(e)
    def one(item):
        folder,es=item
        dest=DATA/'poses'/(folder.split('/')[-1]+'.gz')
        if dest.exists(): return {'name':folder,'ok':True,'frames':len(es),'cached':True}
        lo=min(e.header_offset for e in es)
        hi=max(e.header_offset+30+len(e.filename.encode())+65535+e.compress_size for e in es)
        hi=min(hi,SIZE)
        blob=get(lo,hi-1)
        rows=[]
        for e in es:
            off=e.header_offset-lo
            h=struct.unpack_from('<4sHHHHHIIIHH',blob,off)
            start=off+30+h[-2]+h[-1]
            c=blob[start:start+e.compress_size]
            if e.compress_type==9: b=Inflater().inflate(c)
            elif e.compress_type==8: b=zlib.decompress(c,-15)
            elif e.compress_type==0: b=c
            else: raise ValueError('Unsupported pose compression')
            if len(b)!=e.file_size or zlib.crc32(b)&0xffffffff!=e.CRC: raise ValueError('Pose CRC mismatch')
            rows.append({'frame':int(re.search(r'_(\d+)_keypoints',e.filename).group(1)),
                         'sha256':hashlib.sha256(b).hexdigest(),'data':json.loads(b)})
        rows.sort(key=lambda r:r['frame'])
        dest.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(dest,'wt',encoding='utf-8') as f: json.dump(rows,f,separators=(',',':'))
        return {'name':folder,'ok':True,'frames':len(es)}
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fs={pool.submit(one,item):item[0] for item in groups.items()}
        for f in concurrent.futures.as_completed(fs):
            try: out.append(f.result())
            except Exception as ex: out.append({'name':fs[f],'ok':False,'error':str(ex)})
            if len(out)%50==0: print(f'poses {len(out)}/{len(groups)}; failed={sum(not x["ok"] for x in out)}',flush=True)
    (HERE/'download-poses.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'total':len(out),'ok':sum(x['ok'] for x in out)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');a=p.parse_args()
    if a.action=='inventory': inventory()
    elif a.action=='poses': poses()
    else: download(a.action)
