"""Audit public sample before acquiring any large multi-part archive."""
from pathlib import Path
import requests,json,hashlib,zipfile,shutil
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104/healthgait')
def main():
    BIG.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(BIG).free>5*1024**3
    r=requests.get('https://zenodo.org/api/records/14039922',timeout=60);r.raise_for_status();meta=r.json()
    (BIG/'record.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    f=next(x for x in meta['files'] if x['key']=='dataset_samples.zip');p=BIG/f['key']
    if not p.exists():
        with requests.get(f['links']['self'],stream=True,timeout=60) as r:
            r.raise_for_status()
            with p.open('wb') as out:
                for block in r.iter_content(1024*1024):out.write(block)
    assert p.stat().st_size==f['size'] and hashlib.md5(p.read_bytes()).hexdigest()==f['checksum'].split(':')[1]
    with zipfile.ZipFile(p) as z:
        inv=[{'name':x.filename,'bytes':x.file_size} for x in z.infolist()]
        for x in z.infolist():
            if x.filename.lower().endswith(('.csv','.txt')):
                q=BIG/'sample-metadata'/Path(x.filename).name;q.parent.mkdir(exist_ok=True);q.write_bytes(z.read(x))
    (HERE/'healthgait-sample-audit.json').write_text(json.dumps({'source':'https://zenodo.org/records/14039922','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'official_md5_verified':True,'members':inv},indent=2),encoding='utf8')
    print('SAMPLE',len(inv),'members',flush=True)
    print(json.dumps([x for x in inv if x['name'].lower().endswith(('.csv','.txt','.mp4','.avi','.mov'))],indent=2),flush=True)
if __name__=='__main__':main()
