from audit import *
import urllib.request,struct,zlib,time
URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip'
def main():
    audit=json.loads((HERE/'audit.json').read_text())
    ids={r['id'] for r in audit if r['target']>=20 or (r['in176'] and any(e['outside_old3_seconds']>=1 for e in r['events']))}
    rows=json.loads((PREV/'common176/manifest.json').read_text());index={x['name']:x for x in json.loads((HERE.parent/'20261030-opencap415-tug-timeseries/archive-index.json').read_text())}
    wanted=[]
    for m in rows:
        if m['id'] not in ids:continue
        name=m['turn_file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc');wanted.append(name)
        parent=Path(m['turn_file']).parent.parent.as_posix()+'/'
        wanted.extend(n for n in index if n.startswith(parent) and n.endswith('.yaml'))
    receipts=[]
    for name in dict.fromkeys(wanted):
        x=index[name];p=HERE/'raw'/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            start=x['offset'];end=start+x['compressed']+1023
            for attempt in range(3):
                try:
                    req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=45) as r:
                        assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-');raw=r.read()
                    assert raw[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',raw,26);off=30+nl+el
                    assert raw[30:30+nl].decode()==name
                    b=zlib.decompress(raw[off:off+x['compressed']],-15)
                    assert len(b)==x['size'] and zlib.crc32(b)==x['crc'];p.write_bytes(b);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2)
        b=p.read_bytes();assert len(b)==x['size'] and zlib.crc32(b)==x['crc']
        receipts.append({'name':name,'bytes':len(b),'sha256':sha(p),'crc_verified':True})
        save('acquisition.json',receipts);print('ACQUIRED',Path(name).name,flush=True)
    print('PEOPLE',len(ids),'FILES',len(receipts),flush=True)
if __name__=='__main__':main()
