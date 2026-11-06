"""Public CC0 video archive, local only; verify publisher MD5 and every extracted CRC."""
from pathlib import Path
import json,hashlib,urllib.request,zipfile,shutil,time,re
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104')
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def digest(p,kind):
    h=hashlib.new(kind)
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    meta=json.loads(Path('D:/gait-research-data/D_P/outputs/gait-public-access-20261015/toronto-19929893.json').read_text());assert meta['license']['name']=='CC0';source=meta['files'][0];assert source['id']==35454242 and source['size']==2351609541
    raw=B/'raw';raw.mkdir(exist_ok=True);archive=raw/'Toronto-Videos-35454242.zip';part=archive.with_suffix('.zip.part');outdir=B/'videos';outdir.mkdir(exist_ok=True)
    assert shutil.disk_usage(B).free>6_000_000_000,'Need planned<=6GB additional local disk capacity'
    plan=dict(source=source['download_url'],expected_bytes=source['size'],expected_publisher_md5=source['computed_md5'],license=meta['license'],scope='Public fourteen bottom-camera videos, no external upload. Download complete2.35GB archive to verify full publisherMD5; extract only fourteen bottom videos~1.25GB, reuse original pilot ifCRCmatches. Keep all14subjects independent of scores; camera/ID/time audit before further extraction or supervised fit.',maximum_planned_additional_GB=6)
    if(raw/'download-plan.json').exists():assert json.loads((raw/'download-plan.json').read_text())==plan
    else:save(raw/'download-plan.json',plan)
    if not archive.exists():
        offset=part.stat().st_size if part.exists() else 0;headers={'Range':f'bytes={offset}-'} if offset else {};request=urllib.request.Request(source['download_url'],headers=headers);start=time.monotonic()
        with urllib.request.urlopen(request,timeout=90) as response:
            assert response.status==(206 if offset else 200),response.status
            if offset:assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
            total=offset;last_print=total
            with part.open('ab' if offset else 'wb') as f:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    total+=len(chunk);assert total<=source['size'];f.write(chunk)
                    if total-last_print>=256*1024*1024:print('archive bytes',total,'elapsed',round(time.monotonic()-start,1),flush=True);last_print=total
        assert total==source['size'];assert digest(part,'md5')==source['computed_md5'];part.replace(archive)
    assert archive.stat().st_size==source['size'] and digest(archive,'md5')==source['computed_md5'];archive_sha=digest(archive,'sha256');rows=[]
    with zipfile.ZipFile(archive) as z:
        members=sorted([q for q in z.infolist() if re.fullmatch(r'Videos/OAW\d{2}-bottom\.mp4',q.filename)],key=lambda q:q.filename);assert len(members)==14
        for q in members:
            target=outdir/Path(q.filename).name;pilot=B/Path(q.filename).name
            if pilot.exists():target=pilot
            if not target.exists():
                temp=target.with_suffix('.mp4.part')
                with z.open(q) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                assert temp.stat().st_size==q.file_size;temp.replace(target)
            import zlib
            crc=0
            with target.open('rb') as src:
                for chunk in iter(lambda:src.read(1024*1024),b''):crc=zlib.crc32(chunk,crc)
            assert crc==q.CRC;rows.append(dict(person_id=Path(q.filename).name.split('-')[0],file=str(target),member=q.filename,bytes=q.file_size,crc32=crc,crc_verified=True,sha256=digest(target,'sha256')));print('verified video',rows[-1]['person_id'],flush=True)
    labels=json.loads((H/'clinical-label-audit.json').read_text());assert set(r['person_id'] for r in rows)==set(r['person_id'] for r in labels['labels'])
    save(H/'all-bottom-videos-manifest.json',dict(status='source_files_verified_pending_camera_review',people=14,source=plan['source'],license=meta['license'],whole_archive_md5_verified=True,whole_archive_sha256=archive_sha,archive=str(archive),rows=rows,clinical_label_ids_match=True,external_upload=False));print('14realTUGvideo sources verified',flush=True)
if __name__=='__main__':main()
