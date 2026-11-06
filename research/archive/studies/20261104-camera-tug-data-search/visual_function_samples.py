"""Small original pose and video-directory pilots, never full video archives."""
from pathlib import Path
import json,csv,hashlib,requests,urllib.parse,sys,io,zipfile,shutil
H=Path('D:/Projects/D_P/studies/20261104-camera-tug-data-search')
B=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def get(url,headers=None,cap=10*1024**2):
    with requests.get(url,headers=headers,stream=True,timeout=40) as r:
        r.raise_for_status();blocks=[];total=0
        for block in r.iter_content(65536):
            total+=len(block)
            if total>cap:raise ValueError('pilotcap exceeded')
            blocks.append(block)
        return r.status_code,r.headers,b''.join(blocks)
class Remote(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.network=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,w=0):self.pos=n if w==0 else self.pos+n if w==1 else self.size+n;return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n<=0:return b''
        assert n<4*1024**2 and self.network+n<8*1024**2
        end=self.pos+n-1;status,headers,body=get(self.url,{'Range':f'bytes={self.pos}-{end}'},cap=4*1024**2)
        assert status==206 and headers['Content-Range'].startswith(f'bytes {self.pos}-{end}/') and len(body)==n
        self.pos+=n;self.network+=n;return body
def schema(x):
    if isinstance(x,dict):return {'type':'object','keys':list(x)[:25],'sample_value_types':{k:type(v).__name__ for k,v in list(x.items())[:10]}}
    if isinstance(x,list):return {'type':'list','length':len(x),'first_record':schema(x[0]) if x else None}
    return {'type':type(x).__name__}
def main():
    report={};assert shutil.disk_usage(B).free>2*1024**3
    inv=json.loads((H/'file-audit.json').read_text(encoding='utf8'))['mmu132']['files'];sizes={r['name']:r['totalBytes'] for r in inv}
    rows=list(csv.DictReader((B/'mmu132-final_gait_feature.csv.download').open(encoding='utf-8-sig')))
    ids=[next(r['ID'] for r in rows if float(r['age'])>=65),next(r['ID'] for r in rows if float(r['age'])<65)];pilot=[]
    for pid in ids:
      for mode in ['2D','3D']:
        name=f'MMU-FRiP/Raw/{mode}/{pid}_{mode}/{pid}_{mode}.json';assert sizes[name]<10*1024**2
        url='https://www.kaggle.com/api/v1/datasets/download/limzhekhae1008/mmu-frip/'+urllib.parse.quote(name,safe='')
        _,headers,data=get(url);assert len(data)==sizes[name];x=json.loads(data)
        dest=B/'mmu-pose-pilot'/Path(name).name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
        pilot.append({'id':pid,'mode':mode,'bytes':len(data),'sha256':sha(data),'schema':schema(x),'path':str(dest)})
        print('POSE_READ',pid,mode,flush=True)
    report['MMU_pose_pilot']=pilot
    text=(B/'KINECAL_register.metadata').read_text(encoding='utf-8-sig');rr=list(csv.DictReader(io.StringIO(text)));report['KINECAL']={'rows':len(rr),'fields':list(rr[0]),'continuous_TUG_or_SPPB_column_present':any('tug' in k.lower() or 'sppb' in k.lower() for k in rr[0])}
    p=B/'PDFE-PDFEinfo.csv';data=p.read_bytes()
    for enc in ['utf-8-sig','cp1252','latin1']:
        try:text=data.decode(enc);break
        except UnicodeDecodeError:pass
    dialect=csv.Sniffer().sniff(text[:4096],delimiters=',;\t');rr=list(csv.DictReader(io.StringIO(text),dialect=dialect));cols=list(rr[0]);report['PDFE_metadata']={'rows':len(rr),'encoding':enc,'delimiter':dialect.delimiter,'columns':cols,'sha256':sha(data)}
    # No trial data are executed. ZIP directory names only, streamed through HTTP ranges.
    meta=json.loads((B/'PDFE_figshare.metadata').read_text());f=next(x for x in meta['files'] if x['name']=='Videos.zip');remote=Remote(f['download_url'],f['size']);z=zipfile.ZipFile(remote)
    names=[{'name':x.filename,'bytes':x.file_size,'compressed':x.compress_size,'offset':x.header_offset,'crc32':x.CRC} for x in z.infolist()]
    (B/'PDFE-video-inventory.json').write_text(json.dumps(names,indent=2),encoding='utf8')
    report['PDFE_video_inventory']={'members':len(names),'extensions':dict(__import__('collections').Counter(Path(x['name']).suffix.lower() for x in names)),'examples':names[:8],'range_network_bytes':remote.network,'no_video_payload_downloaded':True}
    (H/'visual-function-samples.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
