from common import *
import urllib.request,struct,zlib,time
URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip'

def main():
    pairs=pd.read_csv(HERE/'selected-pairs.csv');ids=sorted(set(pairs.id_a)|set(pairs.id_b))
    manifest=pd.read_csv(PRIOR/'STS_common6/manifest.csv').set_index('id')
    index=json.loads((SOURCE/'archive-index.json').read_text());lookup={r['name']:r for r in index};wanted=[]
    for pid in ids:
        r=manifest.loc[pid];marker=r['file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc')
        if marker in lookup:wanted.append(lookup[marker])
        prefix=f'/{pid}/visit-{int(r.visit)}/'
        wanted.extend(a for a in index if prefix in a['name'] and a['name'].endswith('.mot') and 'xsts' in a['name'])
    save('acquisition-plan.json',{'source':URL,'files':len(wanted),'bytes':sum(x['size'] for x in wanted),'entries':wanted,
                                 'purpose':'walking markers for same6s quality; full chair signals only to audit outcome provenance, never predictor inputs'})
    receipts=[]
    for r in wanted:
        dst=HERE/'raw'/r['name'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            start=r['offset'];end=start+r['compressed']+1023
            for attempt in range(3):
                try:
                    req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=30) as response:
                        assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {start}-');raw=response.read()
                    assert raw[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',raw,26);offset=30+nl+el
                    assert raw[30:30+nl].decode()==r['name'];b=zlib.decompress(raw[offset:offset+r['compressed']],-15)
                    assert len(b)==r['size'] and zlib.crc32(b)==r['crc'];dst.write_bytes(b);break
                except Exception as e:
                    if attempt==2:raise
                    time.sleep(1)
        b=dst.read_bytes();assert len(b)==r['size'] and zlib.crc32(b)==r['crc']
        receipts.append({'name':r['name'],'bytes':len(b),'sha256':sha(dst),'crc_verified':True})
        save('acquisition.json',receipts);print('ACQUIRED',len(receipts),'/',len(wanted),dst.name,flush=True)

if __name__=='__main__':main()
