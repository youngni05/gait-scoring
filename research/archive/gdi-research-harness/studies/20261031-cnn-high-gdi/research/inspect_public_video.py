"""Read the public ZIP directory using HTTP ranges; no participant video upload."""
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import time
import http.client

ROOT=Path(__file__).resolve().parent
URL='https://zenodo.org/records/21278362/files/VIDEO.zip?download=1'
def fetch(byte_range):
    req=urllib.request.Request(URL,headers={'Range':'bytes='+byte_range})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req,timeout=60) as r:
                if r.status!=206:raise ValueError('Server did not honor range')
                content_range=r.headers['Content-Range']
                value=r.read()
            return value,content_range
        except (OSError,http.client.IncompleteRead):
            if attempt==3:raise
            time.sleep(2)
def main():
    tail,cr=fetch('-524288')
    start=int(cr.split()[1].split('-')[0])
    (ROOT/'video_zip_directory.bin').write_bytes(tail)
    with zipfile.ZipFile(io.BytesIO(tail)) as z:
        rows=[dict(name=i.filename,size=i.file_size,compressed_size=i.compress_size,
                   header_offset=i.header_offset+start,crc=i.CRC,compression=i.compress_type)
              for i in z.infolist()]
    (ROOT/'video_inventory.json').write_text(json.dumps({'url':URL,'range':cr,'files':rows},indent=2),encoding='utf-8')
    normal=[r for r in rows if '/TD/' in r['name'] or '/S01/' in r['name']]
    print(json.dumps({'entries':len(rows),'normal_entries':len(normal),'normal_compressed_bytes':sum(r['compressed_size'] for r in normal),'sample':normal[:12],'first':rows[:3]},ensure_ascii=False))
if __name__=='__main__':main()
