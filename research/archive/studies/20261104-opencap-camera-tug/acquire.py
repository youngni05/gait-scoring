"""Acquire only the frozen cohort's public marker files; verify member CRCs."""
from pathlib import Path
import concurrent.futures as cf
import csv, hashlib, json, struct, time, zlib
import requests

HERE=Path('D:/Projects/D_P/studies/20261104-opencap-camera-tug')
HEAVY=Path('D:/gait-research-data/D_P/opencap-camera-tug-20261104')
OLD=Path('D:/Projects/D_P/gdi-research-harness/studies/20261030-opencap415-tug-timeseries')
STABILITY=OLD.parent/'20261030-opencap415-speed-stability'
URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip'

def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    HERE.mkdir(parents=True,exist_ok=True); HEAVY.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((STABILITY/'manifest.csv').open()))
    index={x['name']:x for x in json.loads((OLD/'archive-index.json').read_text())}
    current=json.loads(Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104/nmd415-zip-inventory.json').read_text())
    if isinstance(current,dict):
        print('INVENTORY_KEYS',list(current),flush=True)
        current=current.get('members',current.get('entries',current.get('files')))
    ci={x['name']:x for x in current}
    selected=[]
    for row in rows:
        name=row['file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc')
        x=index[name]; assert (x['size'],x['compressed'])==(ci[name]['size'],ci[name]['compressed'])
        selected.append(x)
    (HERE/'acquisition-plan.json').write_text(json.dumps({'url':URL,'member_count':len(selected),
        'uncompressed_bytes':sum(x['size'] for x in selected),'compressed_bytes':sum(x['compressed'] for x in selected),
        'archive_index_sha256':sha((OLD/'archive-index.json').read_bytes()),'whole_archive_downloaded':False,
        'members':selected},indent=2))
    def fetch(x):
        dst=HEAVY/'raw'/x['name']; reused=False
        if dst.exists():
            b=dst.read_bytes(); assert len(b)==x['size'] and zlib.crc32(b)==x['crc']; reused=True
        else:
            pilot=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104/opencap-file-pilot')/Path(x['name']).name
            if pilot.exists():
                b=pilot.read_bytes(); assert len(b)==x['size'] and zlib.crc32(b)==x['crc']; reused=True
            else:
                start=x['offset'];end=start+x['compressed']+1023
                for attempt in range(5):
                    try:
                        r=requests.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=90)
                        assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                        a=r.content;assert a[:4]==b'PK\x03\x04'
                        method=struct.unpack_from('<H',a,8)[0]; assert method==8
                        nl,el=struct.unpack_from('<HH',a,26);off=30+nl+el
                        assert a[30:30+nl].decode('utf-8')==x['name']
                        b=zlib.decompress(a[off:off+x['compressed']],-15)
                        assert len(b)==x['size'] and zlib.crc32(b)==x['crc'];break
                    except Exception:
                        if attempt==4:raise
                        time.sleep(3*(attempt+1))
            dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b)
        return {'member':x['name'],'bytes':len(b),'sha256':sha(b),'crc32':x['crc'],'crc_verified':True,'reused':reused}
    out=[]
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        for item in pool.map(fetch,selected):
            out.append(item)
            (HERE/'acquisition.json').write_text(json.dumps({'complete':len(out)==len(selected),'members':out},indent=2))
            if len(out)%20==0: print('VERIFIED',len(out),'/',len(selected),flush=True)
    print('ACQUIRED',len(out),'FILES',sum(x['bytes'] for x in out),'BYTES',flush=True)
if __name__=='__main__':main()
