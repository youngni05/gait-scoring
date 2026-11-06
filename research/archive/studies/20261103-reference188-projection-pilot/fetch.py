import sys,json,zipfile,struct,urllib.request,time,zlib,hashlib
from pathlib import Path
BASE=Path("D:/Projects/D_P/studies/20261103-reference188-pilot")
HERE=Path(__file__).resolve().parent
BIG=Path("D:/gait-research-data/D_P/dataset-audits/20261103-reference188-projection-pilot")
sys.path.insert(0,str(BASE));from inspect_mat import Remote
BIG.mkdir(parents=True,exist_ok=True)
selection=[(42450060,463159769,"138_HealthyPiG_10.05/SUBJ%02d/SUBJ%d (1).c3d"%(i,i),"SUBJ%02d"%i) for i in [1,2,3]]+[(42450069,316448944,"50_StrokePiG/%s/%s"%(s,t),s) for s,t in [("TVC04","BWA_6.c3d"),("TVC08","BWA1.c3d"),("TVC09","BWA1.c3d")]]
(HERE/"selection.json").write_text(json.dumps(dict(rule="First dynamic trial by original ID; 3 healthy older adults and 3 stroke older adults; selection before predictions",members=selection),indent=2))
manifest=[]
for fid,size,name,person in selection:
 p=BIG/(person+".c3d")
 with zipfile.ZipFile(Remote(fid,size)) as z:
  i=z.getinfo(name)
  if not p.exists():
   r=z.fp;r.seek(i.header_offset);h=r.read(30);nl,el=struct.unpack("<HH",h[26:30]);start=i.header_offset+30+nl+el;end=start+i.compress_size-1
   q=urllib.request.Request("https://ndownloader.figshare.com/files/%d?pilot=%d"%(fid,time.time_ns()),headers={"Range":"bytes=%d-%d"%(start,end)})
   with urllib.request.urlopen(q,timeout=120) as response:
    assert response.status==206 and response.headers["Content-Range"]=="bytes %d-%d/%d"%(start,end,size)
    compressed=response.read()
   assert len(compressed)==i.compress_size
   data=zlib.decompress(compressed,-15) if i.compress_type==8 else compressed
   assert len(data)==i.file_size and zlib.crc32(data)==i.CRC
   p.write_bytes(data)
  data=p.read_bytes();assert len(data)==i.file_size and zlib.crc32(data)==i.CRC
  manifest.append(dict(person=person,source_member=name,file_id=fid,path=str(p),bytes=len(data),crc32=hex(i.CRC),sha256=hashlib.sha256(data).hexdigest()))
 (HERE/"download-manifest.json").write_text(json.dumps(manifest,indent=2));print(person,"verified",len(data),flush=True)
