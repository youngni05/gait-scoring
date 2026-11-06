"""Pinned public release; CRC, expected member sizes and prior pilot hashes."""
from pathlib import Path
import requests,json,hashlib,zipfile,csv,shutil,time
H=Path(__file__).resolve().parent
B=Path('D:/gait-research-data/D_P/mmu-frip-evaluation-20261105')
OLDH=H.parent/'20261104-camera-tug-data-search'
OLDB=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    B.mkdir(exist_ok=True);assert shutil.disk_usage(B).free>3*1024**3
    old=json.loads((OLDH/'file-audit.json').read_text(encoding='utf8'))['mmu132'];expected={r['name']:r['totalBytes'] for r in old['files']}
    p=B/'mmu-frip-v3.zip';url='https://www.kaggle.com/api/v1/datasets/download/limzhekhae1008/mmu-frip?datasetVersionNumber=3'
    if not p.exists():
      for attempt in range(3):
        try:
          with requests.get(url,stream=True,timeout=60) as r:
            r.raise_for_status();n=0;last=0
            with p.with_suffix('.partial').open('wb') as f:
              for chunk in r.iter_content(1024*1024):
                n+=len(chunk);assert n<1024**3;f.write(chunk)
                if n-last>=25*1024**2:print('DOWNLOAD_MB',n//1024**2,flush=True);last=n
          p.with_suffix('.partial').replace(p);break
        except Exception:
          if attempt==2:raise
          time.sleep(2)
    records=[]
    with zipfile.ZipFile(p) as z:
      info={x.filename:x for x in z.infolist() if not x.is_dir()};assert set(expected)==set(info),(len(expected),len(info))
      for name,size in expected.items():assert info[name].file_size==size
      assert z.testzip() is None
      table='MMU-FRiP/Extracted/final_gait_feature.csv';readme='MMU-FRiP/README.md'
      assert hashlib.sha256(z.read(table)).hexdigest()==sha(OLDB/'mmu132-final_gait_feature.csv.download')
      assert hashlib.sha256(z.read(readme)).hexdigest()==sha(OLDB/'mmu132-README.md.download')
      rows=list(csv.DictReader(z.read(table).decode('utf-8-sig').splitlines()));ids={r['ID'] for r in rows};assert len(ids)==125
      for name,x in info.items():
        if '/Raw/' in name and Path(name).name.split('_')[0] not in ids:continue
        dest=B/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name));records.append({'member':name,'path':str(dest),'bytes':x.file_size,'crc32':x.CRC,'sha256':sha(dest)})
    prior=json.loads((OLDH/'visual-function-samples.json').read_text(encoding='utf8'))['MMU_pose_pilot']
    for r in prior:
        dest=B/'source/MMU-FRiP/Raw'/r['mode']/f'{r["id"]}_{r["mode"]}'/f'{r["id"]}_{r["mode"]}.json'
        assert sha(dest)==r['sha256']
    (H/'acquisition.json').write_text(json.dumps({'version':3,'url':url,'archive_sha256':sha(p),'archive_bytes':p.stat().st_size,'all267_members_CRC_verified':True,'people_with_scores':125,'members':records},indent=2),encoding='utf8')
    print('ACQUIRED',len(records),'MEMBERS',125,'PEOPLE',p.stat().st_size,'ZIP_BYTES',flush=True)
if __name__=='__main__':main()
