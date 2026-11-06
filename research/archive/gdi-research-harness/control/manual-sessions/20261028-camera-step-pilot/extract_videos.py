import sys,json,struct,zlib,hashlib
from pathlib import Path
S=Path(__file__).parent
rows=json.loads((S.parent.parent/'desktop-sessions/20261028T042859Z-a7273a/walking_pair_candidates.json').read_text(encoding='utf-8-sig'))[0]['members']
out=[]
for row in rows:
 cam=Path(row['name']).stem[-2:]; b=(S/f'camera{cam}.zipsegment').read_bytes(); h=struct.unpack('<4s5H3I2H',b[:30]); assert h[0]==b'PK\x03\x04'; start=30+h[-2]+h[-1]
 data=zlib.decompress(b[start:start+row['compressed_bytes']],-15)
 assert len(data)==row['uncompressed_bytes'] and zlib.crc32(data)==row['crc32']
 p=S/f'camera{cam}.mp4'; p.write_bytes(data)
 out.append(dict(camera=cam,member=row['name'],bytes=len(data),crc32_verified=True,sha256=hashlib.sha256(data).hexdigest(),source='https://www.tnt.uni-hannover.de/project/TNT15/',use='research only; no product redistribution'))
(S/'video_manifest.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('verified videos:',len(out))
