from pathlib import Path
import json,urllib.request,struct,zlib,hashlib,time,shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip'
def save(n,x):(HERE/n).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8')
def main():
    index=json.loads((ROOT/'20261030-opencap415-tug-timeseries/archive-index.json').read_text())
    rows=json.loads((ROOT/'20261031-opencap-preparation-offset/include283/manifest.json').read_text())
    wanted=[]
    for m in rows:
        if m['id'] not in ['sub-283','sub-288']:continue
        parent=Path(m['turn_file']).parent.parent.as_posix()+'/'
        wanted.extend(x for x in index if x['name'].startswith(parent) and (x['name'].endswith('.yaml') or ('task-tug_' in x['name'] and x['name'].endswith('.trc'))))
    receipts=[]
    for x in wanted:
        dst=HERE/'raw'/x['name'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            previous=ROOT/'20261031-opencap-delay/raw'/x['name']
            if previous.exists():shutil.copyfile(previous,dst)
            else:
                start=x['offset'];end=start+x['compressed']+1023
                for attempt in range(3):
                    try:
                        req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                        with urllib.request.urlopen(req,timeout=30) as r:
                            assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-');raw=r.read()
                        assert raw[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',raw,26);off=30+nl+el;assert raw[30:30+nl].decode()==x['name']
                        b=zlib.decompress(raw[off:off+x['compressed']],-15);assert len(b)==x['size'] and zlib.crc32(b)==x['crc'];dst.write_bytes(b);break
                    except Exception:
                        if attempt==2:raise
                        time.sleep(1)
        b=dst.read_bytes();assert len(b)==x['size'] and zlib.crc32(b)==x['crc']
        receipts.append({'name':x['name'],'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'crc_verified':True})
        save('acquisition.json',receipts);print('ACQUIRED',dst.name,flush=True)
    for name,url in {'author-tug.py':'https://raw.githubusercontent.com/stanfordnmbl/opencap-fshd-dm-analysis/main/feature_extraction/featurize_tug.py','author-tree.json':'https://api.github.com/repos/stanfordnmbl/NMD_Detection_Quantification/git/trees/main?recursive=1','author-readme.md':'https://raw.githubusercontent.com/stanfordnmbl/NMD_Detection_Quantification/main/README.md'}.items():
        with urllib.request.urlopen(url,timeout=30) as r:b=r.read()
        (HERE/name).write_bytes(b);receipts.append({'name':name,'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});save('acquisition.json',receipts)
    print('COMPLETE',len(receipts),flush=True)
if __name__=='__main__':main()
