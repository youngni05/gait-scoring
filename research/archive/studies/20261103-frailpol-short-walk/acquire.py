from pathlib import Path
import urllib.request,json,hashlib,zipfile,sys,time
sys.stdout.reconfigure(encoding='utf8')
HERE=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk')
BIG=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103')
def main():
    HERE.mkdir(parents=True,exist_ok=True);(BIG/'raw').mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://api.figshare.com/v2/articles/29315771',timeout=60) as r:meta=json.load(r)
    (HERE/'source-record.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    records=[]
    for f in meta['files']:
        if f['name']=='plots.zip':continue
        path=BIG/'raw'/f['name']
        if not path.exists() or path.stat().st_size!=f['size']:
            temp=path.with_suffix(path.suffix+'.part')
            url=f['download_url']+'?download=1&request_time='+str(int(time.time()))
            with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as dest:
                total=0
                while chunk:=response.read(4*1024*1024):
                    dest.write(chunk);total+=len(chunk)
                    if total%(32*1024*1024)==0:print(f['name'],total,'bytes received',flush=True)
            assert temp.stat().st_size==f['size']
            temp.replace(path)
        data=path.read_bytes();md5=hashlib.md5(data).hexdigest();expected=f.get('computed_md5') or f.get('supplied_md5');assert md5==expected,(f['name'],md5,expected)
        records.append(dict(name=f['name'],id=f['id'],path=str(path),bytes=len(data),url=f['download_url'],md5=md5,sha256=hashlib.sha256(data).hexdigest()))
        print(f['name'],len(data),'verified',flush=True)
    (HERE/'acquisition.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    with zipfile.ZipFile(BIG/'raw/IMU_signals.zip') as z:
        inventory=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size) for i in z.infolist()]
        (HERE/'archive-inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf8')
        print('ZIP files',len(inventory),'uncompressed bytes',sum(i['bytes'] for i in inventory),flush=True)
        print(inventory[:12])
if __name__=='__main__':main()
