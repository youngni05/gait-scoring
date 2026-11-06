"""Download one complete, independently compressed video per severity group."""
import concurrent.futures, hashlib, json, pathlib, struct, subprocess, sys, urllib.request, zlib
HERE=pathlib.Path(__file__).resolve().parent
DATA=pathlib.Path('D:/gait-research-data/D_P/dataset-audits/20261102-sideview-reference')
sys.path.insert(0,str(HERE/'deps'))
import rarfile
url=(HERE/'koa-range-url.txt').read_text(encoding='utf-8-sig').strip()
records=json.loads((HERE/'koa-rar-inventory.json').read_text())['entries']
selected={}
for r in records:
    if r['filename'].lower().endswith('.mov'):
        selected.setdefault(str(pathlib.PurePosixPath(r['filename']).parent),r)
prefix=(DATA/'koa-rar-header.bin').read_bytes()[:29]
endbody=b'\x03\x05\x00\x00'
end=struct.pack('<I',zlib.crc32(endbody))+endbody
outdir=DATA/'samples'
outdir.mkdir(exist_ok=True)
def get(r):
    name=pathlib.PurePosixPath(r['filename'])
    assert not name.is_absolute() and '..' not in name.parts
    a=r['header_offset'];b=r['data_offset']+r['compress_size']-1
    req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
    with urllib.request.urlopen(req,timeout=120) as response:
        assert response.status==206
        assert response.headers['Content-Range'].startswith(f'bytes {a}-')
        block=response.read()
    assert len(block)==b-a+1
    archive=outdir/(name.stem+'.rar')
    archive.write_bytes(prefix+block+end)
    rf=rarfile.RarFile(archive)
    assert len(rf.infolist())==1 and not rf.is_solid()
    proc=subprocess.run(['C:/Windows/System32/tar.exe','-xf',str(archive),'-C',str(outdir)],capture_output=True,text=True)
    target=outdir/pathlib.Path(*name.parts)
    raw=target.read_bytes()
    assert len(raw)==r['file_size'] and zlib.crc32(raw)==r['CRC']
    result={'filename':r['filename'],'path':str(target),'size':len(raw),'crc32_verified':True,'sha256':hashlib.sha256(raw).hexdigest(),'extract_exit':proc.returncode,'extract_stderr':proc.stderr}
    print('verified',name.stem,len(raw),flush=True)
    return result
with concurrent.futures.ThreadPoolExecutor(3) as pool: results=list(pool.map(get,selected.values()))
(HERE/'koa-sample-downloads.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
