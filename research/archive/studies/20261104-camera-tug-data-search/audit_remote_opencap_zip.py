from pathlib import Path
import requests,io,zipfile,json,collections,hashlib
H=Path('D:/Projects/D_P/studies/20261104-camera-tug-data-search');B=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104')
URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip?download=1';SIZE=6584902803
class Remote(io.RawIOBase):
    def __init__(self):self.pos=0;self.logs=[];self.total=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,o,whence=0):
        self.pos=o if whence==0 else self.pos+o if whence==1 else SIZE+o;return self.pos
    def read(self,n=-1):
        if n<0:n=SIZE-self.pos
        n=min(n,SIZE-self.pos)
        if n<=0:return b''
        assert n<=8*1024**2 and self.total+n<20*1024**2,'Metadata-only range budget'
        headers={'Range':f'bytes={self.pos}-{self.pos+n-1}','Accept-Encoding':'identity'}
        with requests.get(URL,headers=headers,stream=True,timeout=40) as r:
            assert r.status_code==206,('Server ignored byte range; whole archive not downloaded',r.status_code)
            data=r.raw.read(n+1);assert len(data)==n,(len(data),n)
            self.logs.append(dict(offset=self.pos,bytes=n,status=r.status_code,content_range=r.headers.get('Content-Range'),sha256=hashlib.sha256(data).hexdigest()));self.pos+=n;self.total+=n;return data
r=Remote()
try:
    z=zipfile.ZipFile(r);names=z.namelist();entries=[dict(name=i.filename,size=i.file_size,compressed=i.compress_size) for i in z.infolist()];p=B/'nmd415-zip-inventory.json';p.write_text(json.dumps(entries,indent=2),encoding='utf8');out=dict(status='remote_central_directory_verified',total_members=len(names),extensions=dict(collections.Counter(Path(n).suffix.lower() for n in names if not n.endswith('/'))),video_members=[n for n in names if Path(n).suffix.lower() in ['.mp4','.mov','.avi','.mkv','.wmv']],members_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),inventory=str(p),range_bytes_received=r.total,range_requests=r.logs,source=URL,whole_archive_downloaded=False);print(json.dumps({k:v for k,v in out.items() if k!='range_requests'},indent=2))
except Exception as e:out=dict(status='metadata_range_audit_failed',error=repr(e),range_bytes_received=r.total,range_requests=r.logs,whole_archive_downloaded=False);print(out)
(H/'nmd415-zip-audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
