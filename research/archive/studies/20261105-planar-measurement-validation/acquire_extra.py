"""Acquire one tiny CC-BY processed-table package; inspect another archive directory only."""
import sys,json,hashlib,urllib.request,zipfile,io,struct,zlib
from pathlib import Path
sys.stdout.reconfigure(encoding='utf8')
H=Path(__file__).resolve().parent;B=Path('D:/gait-research-data/D_P/planar-measurement-validation-20261105');META=B/'source-metadata'
def fetch(url,start=None,end=None):
    headers={'User-Agent':'D_P research-source-audit/1.0'}
    if start is not None:headers['Range']=f'bytes={start}-{end}'
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as r:
        if start is not None:
            assert r.status==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'),str((r.status,r.headers.get('Content-Range')))
        data=r.read(2_000_001);assert len(data)<=2_000_000
    return data
def main():
    d=json.loads((META/'pd-change.json').read_text());assert d['metadata']['license']['id']=='cc-by-4.0';item=d['files'][0];assert item['size']<100_000
    payload=fetch(item['links']['self']);assert len(payload)==item['size'] and 'md5:'+hashlib.md5(payload).hexdigest()==item['checksum']
    folder=B/'pd-change';folder.mkdir(exist_ok=True);(folder/item['key']).write_bytes(payload);z=zipfile.ZipFile(io.BytesIO(payload));assert z.testzip() is None
    members=[]
    for entry in z.infolist():
        if entry.is_dir():continue
        target=(folder/entry.filename).resolve();assert target.is_relative_to(folder.resolve());data=z.read(entry);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);members.append({'name':entry.filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    out={'pd_change':{'status':'downloaded_MD5_and_all_ZIP_CRC_passed','license':'CC-BY4.0','archive_sha256':hashlib.sha256(payload).hexdigest(),'members':members,'raw_RGB_included':any(r['name'].lower().endswith(('.mp4','.avi','.mov')) for r in members)},'no_author_code_executed':True}
    item=next(r for r in json.loads((META/'dual21.json').read_text())['files'] if r['key']=='03_markerless_data.zip');size=item['size'];url=item['links']['self']
    try:
        start=size-131072;tail=fetch(url,start,size-1);pos=tail.rfind(b'PK\x05\x06');assert pos>=0;end=struct.unpack_from('<4sHHHHIIH',tail,pos);cdsize,cdoffset=end[5:7];assert cdsize<2_000_000
        cd=tail[cdoffset-start:cdoffset-start+cdsize] if cdoffset>=start else fetch(url,cdoffset,cdoffset+cdsize-1)
        fake=cd+struct.pack('<4sHHHHIIH',b'PK\x05\x06',0,0,end[3],end[4],len(cd),0,0);entries=zipfile.ZipFile(io.BytesIO(fake)).infolist()
        out['dual21_markerless_directory']={'status':'directory_only_not_fullarchive_integrity','archive_bytes':size,'entries':[{'name':r.filename,'size':r.file_size,'compressed':r.compress_size} for r in entries],'raw_RGB_entries':[r.filename for r in entries if r.filename.lower().endswith(('.mp4','.avi','.mov'))]}
    except Exception as exc:out['dual21_markerless_directory']={'status':'failed','reason':str(exc),'no_full_download_attempted':True}
    (H/'extra-acquisition.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False,indent=2)[:10000])

if __name__=='__main__':main()
