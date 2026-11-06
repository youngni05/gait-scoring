"""Bounded public-file inspection. No execution of downloaded code or pickle."""
from pathlib import Path
from urllib.request import Request,urlopen
import io,json,hashlib,tarfile,zipfile,pickletools
import numpy as np
H=Path(__file__).resolve().parent;D=Path('D:/gait-research-data/D_P/synthgait19k-review');D.mkdir(parents=True,exist_ok=True)
requests=[]
def fetch_range(url,start,end):
    req=Request(url+'?download=true&probe_range=%d-%d'%(start,end),headers={'Range':'bytes=%d-%d'%(start,end),'User-Agent':'gdi-research-review'})
    with urlopen(req,timeout=90) as q:
        if q.status!=206:raise RuntimeError('Range unsupported; refusing full download: '+str(q.status))
        cr=q.headers.get('Content-Range');b=q.read(end-start+2)
        assert len(b)<=end-start+1 and cr.startswith('bytes %d-'%start),(cr,len(b))
    requests.append(dict(url=url,start=start,end=end,bytes=len(b),content_range=cr,sha256=hashlib.sha256(b).hexdigest()))
    return b
dataset=json.loads((H/'datasets-metadata.json').read_text());model=json.loads((H/'models-metadata.json').read_text())
url='https://huggingface.co/datasets/SoroushMehraban/SynthGait-19K/resolve/'+dataset['sha']+'/data/train-00000-of-00101.tar'
b=fetch_range(url,0,24*1024**2-1);(D/'train-00000-prefix.bin').write_bytes(b)
samples={};first=None
with tarfile.open(fileobj=io.BytesIO(b),mode='r|') as tf:
    for member in tf:
        if not member.isfile():continue
        stem=Path(member.name).stem
        if first is None:first=stem
        if stem!=first:break
        raw=tf.extractfile(member).read();p=D/Path(member.name).name;p.write_bytes(raw)
        item=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if p.suffix=='.json':item['metadata']=json.loads(raw)
        if p.suffix=='.npz':
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                item['arrays']={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype),values=z[k].tolist() if z[k].size<50 else None) for k in z.files}
        samples[p.suffix]=item
assert {'.mp4','.npz','.json'}<=set(samples)
try:
    import cv2
    cap=cv2.VideoCapture(samples['.mp4']['path']);assert cap.isOpened()
    video=dict(fps=cap.get(cv2.CAP_PROP_FPS),frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ok,frame=cap.read();video['first_frame_decoded']=bool(ok);cap.release()
except ImportError:video=dict(decoded=False,reason='cv2 unavailable; container received, JSON metadata inspected')
# Inspect remote checkpoint's ZIP directory and pickle opcodes, never unpickle.
size=next(v['size'] for v in json.loads((H/'weights-tree.json').read_text()) if v['path']=='gaitxformer.pth.tr')
modelurl='https://huggingface.co/SoroushMehraban/GaitXFormer/resolve/'+model['sha']+'/gaitxformer.pth.tr'
class RemoteIO(io.RawIOBase):
    def __init__(self):self.pos=0;self.total=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=size-self.pos
        if n==0 or self.pos>=size:return b''
        n=min(n,size-self.pos)
        assert self.total+n<8*1024**2,'Checkpoint metadata budget exceeded'
        b=fetch_range(modelurl,self.pos,self.pos+n-1);self.pos+=len(b);self.total+=len(b);return b
remote=RemoteIO()
with zipfile.ZipFile(remote) as z:
    names=z.namelist();pkl=next(n for n in names if n.endswith('/data.pkl') or n=='data.pkl')
    raw=z.read(pkl);(D/'checkpoint-metadata.pkl').write_bytes(raw)
    strings=[arg for op,arg,pos in pickletools.genops(raw) if op.name in ['BINUNICODE','SHORT_BINUNICODE','UNICODE']]
    state_names=[v for v in strings if isinstance(v,str) and (v.startswith('encoder.') or v.startswith('cross_attn_layers.') or v.startswith('gait_heads.') or v=='gait_tokens')]
    checkpoint=dict(full_bytes=size,metadata_bytes_fetched=remote.total,zip_members=len(names),tensor_key_names=len(state_names),key_examples=state_names[:8]+state_names[-12:],pickle_executed=False,full_checkpoint_downloaded=False,pickle_metadata_sha256=hashlib.sha256(raw).hexdigest())
report=dict(passed=True,dataset_revision=dataset['sha'],model_revision=model['sha'],range_requests=requests,samples=samples,video=video,checkpoint=checkpoint,whole_tar_sha256_verified=False,whole_checkpoint_sha256_verified=False,gpu_used=False,downloaded_code_executed=False)
(H/'file-probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(sample=first,video=video,checkpoint=checkpoint),ensure_ascii=False))
