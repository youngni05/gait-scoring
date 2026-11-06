"""Decode every F/PG1 walking frame via AVI index byte ranges; retain source ranges."""
import sys,json,struct,hashlib,concurrent.futures,time
from pathlib import Path
from inspect_sources import HERE,BIG,fetch,dump
sys.path.insert(0,'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path('D:/gait-research-data/MoVi/raw')
provider=json.loads((HERE/'movi-provider-current.json').read_text(encoding='utf-8'))['data']['latestVersion']['files']
files={x['label']:x['dataFile'] for x in provider}
rows=json.loads((HERE/'movi-audit.json').read_text())['subjects']
rows=[r for r in rows if next(v for v in r['views'] if v['camera']=='PG1')['video'] in files]
OUT=BIG/'movi-video-ranges';OUT.mkdir(exist_ok=True)

def check(row):
    sid=row['id'];v=next(v for v in row['views'] if v['camera']=='PG1');f=files[v['video']];local=ROOT/v['video'];url=f"https://borealisdata.ca/api/access/datafile/{f['id']}"
    def get(a,b):
        if local.exists():
            with local.open('rb') as h:h.seek(a);return h.read(b-a+1)
        for attempt in range(3):
            try:return fetch(url,a,b)
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    hpath=OUT/f'{sid:02d}-header.bin';tpath=OUT/f'{sid:02d}-tail.bin'
    if not hpath.exists():hpath.write_bytes(get(0,131071))
    if not tpath.exists():tpath.write_bytes(get(f['filesize']-262144,f['filesize']-1))
    head=hpath.read_bytes();tail=tpath.read_bytes();movi=head.find(b'movi');ix=tail.rfind(b'idx1');assert movi>0 and ix>=0
    size=struct.unpack_from('<I',tail,ix+4)[0];assert ix+8+size<=len(tail)
    entries=[struct.unpack_from('<4sIII',tail,ix+8+i) for i in range(0,size,16)]
    entries=[x for x in entries if x[0]==b'00dc'];a=row['walking_start'];b=row['walking_end'];chosen=entries[a-1:b];assert len(chosen)==b-a+1
    start=movi+chosen[0][2];end=movi+chosen[-1][2]+8+chosen[-1][3]-1
    path=OUT/f'{sid:02d}-walking.bin'
    if not path.exists():path.write_bytes(get(start,end))
    blob=path.read_bytes();assert len(blob)==end-start+1
    best=max(v['intervals'],key=lambda x:x['frames'],default={'start_one_based':a,'end_one_based':b})
    pics={best['start_one_based'],(best['start_one_based']+best['end_one_based'])//2,best['end_one_based']}
    decoded=0
    for frame,e in zip(range(a,b+1),chosen):
        off=movi+e[2]-start;assert blob[off:off+4]==b'00dc';length=struct.unpack_from('<I',blob,off+4)[0];assert length==e[3]
        img=cv2.imdecode(np.frombuffer(blob[off+8:off+8+length],np.uint8),cv2.IMREAD_COLOR);assert img is not None and img.shape==(600,800,3)
        decoded+=1
        if frame in pics:
            dest=HERE/'qa'/f'{sid:02d}-{frame}.jpg';dest.parent.mkdir(exist_ok=True);cv2.imwrite(str(dest),img,[cv2.IMWRITE_JPEG_QUALITY,85])
    result={'id':sid,'video':v['video'],'provider_id':f['id'],'provider_bytes':f['filesize'],'source':'local_original' if local.exists() else 'provider_byte_range',
            'walking_start_one_based':a,'walking_end_one_based':b,'decoded_frames':decoded,'dimensions':[800,600],'walking_range':[start,end],
            'range_sha256':hashlib.sha256(blob).hexdigest(),'full_video_md5_checked':False,'selected_review_frames':sorted(pics),'screen_max_frames':v['max_contiguous_frames']}
    dump(OUT/f'{sid:02d}-verification.json',result)
    print('MoVi video decoded',sid,decoded,flush=True);return result

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(check,rows))
    results.sort(key=lambda r:r['id']);dump(HERE/'movi-video-verification.json',results)
    for page in range(4):
        group=results[page*24:(page+1)*24];sheet=Image.new('RGB',(1280,6*265),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(group):
            frame=r['selected_review_frames'][len(r['selected_review_frames'])//2];p=HERE/'qa'/f"{r['id']:02d}-{frame}.jpg";im=Image.open(p);im.thumbnail((320,240));x=(i%4)*320;y=(i//4)*265;sheet.paste(im,(x,y));draw.text((x+5,y+242),f"ID {r['id']} | frame {frame} | side run {r['screen_max_frames']}",fill='black')
        sheet.save(HERE/'qa'/f'movi-contact-{page+1}.jpg',quality=85)
    print('ALL VERIFIED',len(results),sum(r['decoded_frames'] for r in results))
