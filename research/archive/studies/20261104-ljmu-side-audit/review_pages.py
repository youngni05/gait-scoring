import json
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent
review=json.loads((HERE/'visual-review.json').read_text())
queued=set()
for p in HERE.glob('review-page-*.json'):queued.update(json.loads(p.read_text()))
ps=[p for p in sorted((HERE/'thumbs').glob('*.jpg')) if p.stem not in review and p.stem not in queued]
start=len(list(HERE.glob('review-page-*.json')))+4
for offset in range(0,len(ps),48):
    num=start+offset//48
    page=Image.new('RGB',(1600,1200),'#ddd')
    group=ps[offset:offset+48]
    for j,p in enumerate(group):page.paste(Image.open(p),(j%8*200,j//8*200))
    page.save(HERE/f'contact-{num:02d}.jpg',quality=90)
    (HERE/f'review-page-{num:02d}.json').write_text(json.dumps([p.stem for p in group]))
    print(num,len(group))
