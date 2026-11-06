import sys,json,zipfile,struct,urllib.request,time,zlib,hashlib,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('D:/Projects/D_P/studies/20261103-reference188-pilot')
BIG=Path('D:/gait-research-data/D_P/common-joints-gdi-20261103/audit')
sys.path.insert(0,str(BASE));from inspect_mat import Remote
BIG.mkdir(parents=True,exist_ok=True)
inv=json.loads((BASE/'speed-control/public-file-inventory.json').read_text())
old=json.loads(Path('D:/Projects/D_P/studies/20261103-reference188-projection-pilot/download-manifest.json').read_text())
lookup={r['source_member']:r for r in old}
selected=[]
for group in inv['zip_contents']:
 fid=group['file_id']
 if fid not in [42450060,42450069]:continue
 for e in group['entries']:
  name=e['name'];parts=name.split('/')
  if len(parts)!=3 or parts[1] not in ['SUBJ01','SUBJ02','SUBJ03','TVC04','TVC08','TVC09']:continue
  if Path(name).suffix.lower() not in ['.c3d','.history','.mp','.vsk']:continue
  selected.append(dict(file_id=fid,archive_size=463159769 if fid==42450060 else 316448944,member=name,person=parts[1],expected_bytes=e['bytes']))
(HERE/'selection.json').write_text(json.dumps(selected,indent=2))
print('SELECTED',len(selected),'uncompressed bytes',sum(e['expected_bytes'] for e in selected),flush=True)
manifest=[]
for fid in [42450060,42450069]:
 group=[e for e in selected if e['file_id']==fid]
 with zipfile.ZipFile(Remote(fid,group[0]['archive_size'])) as z:
  for item in group:
   name=item['member'];info=z.getinfo(name);p=Path(lookup[name]['path']) if name in lookup else BIG/item['person']/Path(name).name
   if not p.exists():
    z.fp.seek(info.header_offset);head=z.fp.read(30);nl,el=struct.unpack('<HH',head[26:30]);start=info.header_offset+30+nl+el;end=start+info.compress_size-1
    for attempt in range(3):
     try:
      q=urllib.request.Request('https://ndownloader.figshare.com/files/%d?audit=%d'%(fid,time.time_ns()),headers={'Range':'bytes=%d-%d'%(start,end)})
      with urllib.request.urlopen(q,timeout=90) as r:
       assert r.status==206 and r.headers['Content-Range']=='bytes %d-%d/%d'%(start,end,item['archive_size']);compressed=r.read()
      assert len(compressed)==info.compress_size;data=zlib.decompress(compressed,-15) if info.compress_type==8 else compressed
      assert len(data)==info.file_size and zlib.crc32(data)==info.CRC
      p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);break
     except Exception:
      if attempt==2:raise
   data=p.read_bytes();assert len(data)==info.file_size and zlib.crc32(data)==info.CRC
   manifest.append({**item,'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'crc32':hex(info.CRC)})
   (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2));print(len(manifest),item['person'],Path(name).name,'verified',flush=True)
