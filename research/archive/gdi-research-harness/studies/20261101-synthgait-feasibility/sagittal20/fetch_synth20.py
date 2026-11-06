"""Retrieve 16 train + 4 validation sagittal walks, bounded streaming TAR reads."""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib, json, tarfile, time

CODE = Path('D:/Projects/D_P/gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20')
DATA = Path('D:/gait-research-data/D_P/synthgait19k-review/sagittal20')
REV = '65e7b13ac7917514fc8867e00fb9ca9f5495877a'

def sha(b): return hashlib.sha256(b).hexdigest()
def save(p, value):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False),encoding='utf-8');tmp.replace(p)

class Counted:
    def __init__(self, response): self.response=response;self.count=0;self.hash=hashlib.sha256()
    def read(self, n):
        value=self.response.read(n);self.count+=len(value);self.hash.update(value);return value

def main():
    DATA.mkdir(parents=True, exist_ok=True); CODE.mkdir(parents=True, exist_ok=True)
    manifest=json.loads((CODE.parent/'metadata/dataset_manifest.json').read_text(encoding='utf-8'))
    records=[];receipts=[];started=time.time()
    for split,wanted in [('train',16),('validation',4)]:
        shard=next(s for s in manifest['shards'] if s['split']==split)
        url='https://huggingface.co/datasets/SoroushMehraban/SynthGait-19K/resolve/'+REV+'/'+shard['path']
        maximum=min(shard['size_bytes'],768*1024**2)
        request=Request(url+'?download=true&sagittal20=1',headers={'Range':'bytes=0-%d'%(maximum-1)})
        selected={}
        with urlopen(request,timeout=90) as response:
            assert response.status==206 and response.headers['Content-Range'].startswith('bytes 0-')
            stream=Counted(response)
            with tarfile.open(fileobj=stream, mode='r|') as archive:
                for member in archive:
                    name=Path(member.name)
                    if not member.isfile() or not name.stem.endswith('_saggital'): continue
                    if name.stem not in selected and len(selected)>=wanted: break
                    assert name.suffix in ('.mp4','.npz','.json')
                    raw=archive.extractfile(member).read();dest=DATA/name.name
                    if dest.exists(): assert sha(dest.read_bytes())==sha(raw),dest
                    else: dest.write_bytes(raw)
                    row=selected.setdefault(name.stem,{'sample_id':name.stem,'walk_id':name.stem.rsplit('_',1)[0],
                        'split':split,'source_shard':shard['path'],'files':{}})
                    row['files'][name.suffix]={'path':str(dest),'bytes':len(raw),'sha256':sha(raw),'tar_offset':member.offset_data}
                    if len(selected)==wanted and all(set(r['files'])=={'.mp4','.npz','.json'} for r in selected.values()): break
            assert len(selected)==wanted and all(len(r['files'])==3 for r in selected.values())
            receipts.append({'split':split,'url':url,'content_range':response.headers['Content-Range'],
                'bytes_read':stream.count,'prefix_sha256':stream.hash.hexdigest(),'whole_shard_verified':False,
                'declared_whole_shard_sha256':shard['sha256']})
        for row in selected.values():
            meta=json.loads(Path(row['files']['.json']['path']).read_text(encoding='utf-8'))
            assert meta['split']==split and meta['view']=='sagittal' and meta['sample_id']==row['sample_id']
            row['metadata']=meta;records.append(row)
        save(CODE/'download-manifest.json',{'revision':REV,'records':records,'receipts':receipts,'complete':False})
        print(json.dumps({'split':split,'samples':len(selected),'read_mib':round(stream.count/1024**2,2)}),flush=True)
    assert len(records)==20 and len({r['walk_id'] for r in records})==20
    result={'revision':REV,'records':records,'receipts':receipts,'complete':True,'seconds':time.time()-started,
        'independent_people_known':False,'gpu_used':False,'actual_bytes_saved':sum(f['bytes'] for r in records for f in r['files'].values())}
    save(CODE/'download-manifest.json',result)
    print(json.dumps({'complete':True,'records':20,'actual_bytes_saved':result['actual_bytes_saved'],'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
