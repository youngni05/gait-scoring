from pathlib import Path
import json,urllib.request,struct,zlib,hashlib,shutil,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip'
def main():
    rows=json.loads((ROOT/'20261031-opencap-directional-pace/exclude283/manifest.json').read_text())
    phases={r['id']:r for r in json.loads((ROOT/'20261031-opencap-tail-diagnosis/person-audit.json').read_text())}
    index={r['name']:r for r in json.loads((ROOT/'20261030-opencap415-tug-timeseries/archive-index.json').read_text())}
    wanted=[]
    for m in rows:
        r=phases[m['id']];p=r['phases'];candidate=False
        if 'window' in p['rise'] and r['pre']:candidate|=r['pre'][0]-p['rise']['window'][1]>=.5
        if 'window' in p['sit'] and r['post']:candidate|=p['sit']['window'][0]-r['post'][1]>=.5
        if candidate:wanted.append(index[m['turn_file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc')])
    (HERE/'acquisition-plan.json').write_text(json.dumps({'files':len(wanted),'bytes':sum(r['size'] for r in wanted),'source':URL,'selection':'input-only candidate near-chair leg>=0.5s, no target filter','entries':wanted},indent=2))
    receipts=[]
    for r in wanted:
        dst=HERE/'raw'/r['name'];dst.parent.mkdir(parents=True,exist_ok=True);mode='download'
        if not dst.exists():
            previous=ROOT/'20261031-opencap-delay/raw'/r['name']
            if previous.exists():shutil.copyfile(previous,dst);mode='existing_local'
            else:
                start=r['offset'];end=start+r['compressed']+1023
                for attempt in range(3):
                    try:
                        req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                        with urllib.request.urlopen(req,timeout=30) as response:
                            assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {start}-');raw=response.read()
                        assert raw[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',raw,26);off=30+nl+el;assert raw[30:30+nl].decode()==r['name']
                        b=zlib.decompress(raw[off:off+r['compressed']],-15);assert len(b)==r['size'] and zlib.crc32(b)==r['crc'];dst.write_bytes(b);break
                    except Exception:
                        if attempt==2:raise
                        time.sleep(1)
        b=dst.read_bytes();assert len(b)==r['size'] and zlib.crc32(b)==r['crc']
        receipts.append({'name':r['name'],'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'mode':mode,'crc_verified':True})
        (HERE/'acquisition.json').write_text(json.dumps(receipts,indent=2));print('ACQUIRED',len(receipts),'/',len(wanted),dst.name,flush=True)
if __name__=='__main__':main()
