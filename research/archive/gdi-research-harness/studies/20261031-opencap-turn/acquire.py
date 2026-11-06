from common import *
import urllib.request,struct,zlib,time,concurrent.futures
URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip'
def get(x):
 dst=HERE/'raw'/x['name'];dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() and zlib.crc32(dst.read_bytes())==x['crc']:b=dst.read_bytes()
 else:
  for attempt in range(5):
   try:
    start=x['offset'];end=start+x['compressed']+1023
    req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=45) as r:
     assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-')
     raw=r.read()
    assert raw[:4]==b'PK\x03\x04'
    nl,el=struct.unpack_from('<HH',raw,26);off=30+nl+el
    assert raw[30:30+nl].decode()==x['name']
    b=zlib.decompress(raw[off:off+x['compressed']],-15)
    assert len(b)==x['size'] and zlib.crc32(b)==x['crc']
    dst.write_bytes(b);break
   except Exception:
    if attempt==4:raise
    time.sleep(5*(attempt+1))
  time.sleep(.5)
 return {'name':x['name'],'sha256':hashlib.sha256(b).hexdigest(),'crc':x['crc'],'bytes':len(b)}
def main():
 manifest=pd.read_csv(PREV/'manifest.csv');pairs=set(zip(manifest.id,manifest.visit))
 index=json.loads((SOURCE/'archive-index.json').read_text())
 selected=[x for x in index if re.search(r'_task-tug_(cone|line)\.mot$',x['name']) and identity(x['name'])[:2] in pairs]
 save('selected-files.json',selected)
 print('SELECTED',len(selected),'PEOPLE',len(set(identity(x['name'])[0] for x in selected)),'MB',sum(x['compressed'] for x in selected)/1e6,flush=True)
 receipts=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for r in pool.map(get,selected):
   receipts.append(r);save('acquisition.json',receipts)
   if len(receipts)%20==0:print('DOWNLOADED',len(receipts),'/',len(selected),flush=True)
 print('ACQUISITION_COMPLETE',len(receipts),flush=True)
if __name__=='__main__':main()
