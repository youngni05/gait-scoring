"""Acquire three official raw files only; CRC/size/SHA recorded, no training."""
from pathlib import Path
import json,requests,struct,zlib,hashlib,shutil
HERE=Path(__file__).resolve().parent
SOURCE=Path('D:/gait-research-data/D_P/planar-tug-improvement-20261104/gstride')
DEST=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104/gstride-raw-audit')
def main():
    index=json.loads((SOURCE/'zip-inventory.json').read_text());meta=json.loads((SOURCE/'zenodo-record.json').read_text())
    selected=[r for r in index if r['name'] in ['_IMU/V001.csv','_IMU/V026.csv','_IMU/V138.csv']];assert len(selected)==3
    assert shutil.disk_usage(DEST.parent).free>sum(x['size']+x['compressed'] for x in selected)+1024**3
    DEST.mkdir(exist_ok=True);url=next(x['links']['self'] for x in meta['files'] if x['key'].endswith('.zip'));receipts=[]
    for x in selected:
        path=DEST/Path(x['name']).name
        if path.exists():raw=path.read_bytes()
        else:
            start=x['offset'];end=start+x['compressed']+1023
            r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=60);r.raise_for_status()
            assert r.status_code==206 and len(r.content)==end-start+1 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
            b=r.content;assert b[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',b,26);assert b[30:30+nl].decode()==x['name']
            method=struct.unpack_from('<H',b,8)[0];compressed=b[30+nl+el:30+nl+el+x['compressed']];raw=zlib.decompress(compressed,-15) if method==8 else compressed
            assert len(raw)==x['size'] and zlib.crc32(raw)==x['crc'];path.write_bytes(raw)
        assert len(raw)==x['size'] and zlib.crc32(raw)==x['crc']
        receipts.append({'member':x['name'],'path':str(path),'bytes':len(raw),'crc32':x['crc'],'sha256':hashlib.sha256(raw).hexdigest()})
        print('RAW_FILE_VERIFIED',path.name,len(raw),raw[:160],flush=True)
    (HERE/'raw-audit-acquisition.json').write_text(json.dumps({'source':'https://zenodo.org/records/17052815','members':receipts,'full_archive_downloaded':False},indent=2),encoding='utf8')
if __name__=='__main__':main()
