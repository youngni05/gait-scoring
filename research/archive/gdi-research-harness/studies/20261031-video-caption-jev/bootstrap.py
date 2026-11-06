"""Acquire public model/video inputs into D, never customer credentials."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import urllib.request
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BIG=Path('D:/gait-research-data/D_P/video-caption-jev-20261031')
REPO='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fetch(url,path):
    if path.exists():return
    temp=path.with_suffix(path.suffix+'.part')
    with urllib.request.urlopen(url,timeout=60) as r,temp.open('wb') as f:
        while chunk:=r.read(1024*1024):f.write(chunk)
    temp.replace(path)

def main():
    BIG.mkdir(parents=True,exist_ok=True)
    for folder in ['model','videos','clips','deps','cache','frames']:(BIG/folder).mkdir(exist_ok=True)
    saved=json.loads((HERE/'acquisition.json').read_text()) if (HERE/'acquisition.json').exists() else None
    metadata_url='https://huggingface.co/api/models/'+REPO
    if saved:metadata_url+='/revision/'+saved['revision']
    meta=json.load(urllib.request.urlopen(metadata_url,timeout=30))
    revision=meta['sha']
    receipt={'model':REPO,'revision':revision,'files':{},'videos':[]}
    for item in meta['siblings']:
        name=item['rfilename']
        if '/' in name or not (name.endswith(('.json','.safetensors','.txt','.jinja'))):continue
        path=BIG/'model'/name
        fetch(f'https://huggingface.co/{REPO}/resolve/{revision}/{name}',path)
        if saved and name in saved['files']:
            assert sha(path)==saved['files'][name]['sha256'], 'existing_model_fingerprint_mismatch'
        receipt['files'][name]={'bytes':path.stat().st_size,'sha256':sha(path)}
        print('model',name,path.stat().st_size,flush=True)
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--no-cache-dir',
                    '--target',str(BIG/'deps'),'transformers==4.57.1','huggingface-hub==0.36.0','tokenizers==0.22.1',
                    'safetensors==0.6.2','regex','packaging','filelock','pyyaml','tqdm','requests','fsspec',
                    'num2words','docopt','av==15.1.0','torchvision==0.29.2',
                    'urllib3','charset-normalizer','idna','certifi','colorama'],check=True)
    records=json.loads((ROOT/'gdi-research-harness/studies/20261030-toronto-poma-tug/preparation.json').read_text())['records']
    people=sorted({r['person'] for r in records if r['status']=='included'})
    members=json.loads((ROOT/'outputs/gait-public-access-20261015/Toronto/video-members.json').read_text())
    archive='https://ndownloader.figshare.com/files/35454242'
    for pid in people:
        name=f'OAW{pid:02d}-bottom.mp4';m=next(m for m in members if m['name']=='Videos/'+name)
        path=BIG/'videos'/name
        if not path.exists():
            existing=ROOT/'outputs/gait-public-access-20261015/Toronto'/name
            if existing.exists():
                # Read the existing verified asset; do not duplicate it.
                path=existing
            else:
                offset=m['local_offset']
                request=urllib.request.Request(archive,headers={'Range':f'bytes={offset}-{offset+29}'})
                with urllib.request.urlopen(request,timeout=60) as r:
                    assert r.status==206;header=r.read()
                assert header[:4]==b'PK\x03\x04'
                nl,el=struct.unpack_from('<HH',header,26)
                start=offset+30+nl+el;end=start+m['compressed_size']-1
                request=urllib.request.Request(archive,headers={'Range':f'bytes={start}-{end}'})
                decoder=zlib.decompressobj(-15);crc=0;total=0;received=0
                tmp=path.with_suffix('.part')
                with urllib.request.urlopen(request,timeout=60) as r,tmp.open('wb') as f:
                    assert r.status==206
                    while chunk:=r.read(1024*1024):
                        received+=len(chunk);raw=decoder.decompress(chunk);f.write(raw);crc=zlib.crc32(raw,crc);total+=len(raw)
                    raw=decoder.flush();f.write(raw);crc=zlib.crc32(raw,crc);total+=len(raw)
                assert received==m['compressed_size'] and total==m['size'] and crc==m['crc32'] and decoder.eof
                tmp.replace(path)
        # CRC also checked for an existing file before using it.
        crc=0
        with path.open('rb') as f:
            while chunk:=f.read(1024*1024):crc=zlib.crc32(chunk,crc)
        assert path.stat().st_size==m['size'] and crc==m['crc32']
        receipt['videos'].append({'person':pid,'path':str(path),'sha256':sha(path),'crc32':crc})
        (HERE/'acquisition.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
        print('video',name,path.stat().st_size,flush=True)

if __name__=='__main__':main()
