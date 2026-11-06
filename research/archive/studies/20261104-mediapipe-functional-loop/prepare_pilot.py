"""Anonymous public CC0 Toronto video pilot; local download only, no server upload."""
from pathlib import Path
import sys,json,zipfile,hashlib,shutil,time
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104')
sys.path.insert(0,'D:/gait-research-data/D_P/outputs/raw-video-score-audit-20261031');from inspect_public_files import RemoteZip
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    H.mkdir(exist_ok=True);B.mkdir(exist_ok=True)
    meta=json.loads(Path('D:/gait-research-data/D_P/outputs/gait-public-access-20261015/toronto-19929893.json').read_text());assert meta['license']['name']=='CC0';f=meta['files'][0]
    with zipfile.ZipFile(RemoteZip(f['download_url'],f['size'])) as z:
        candidates=[i for i in z.infolist() if i.filename.endswith('-bottom.mp4')];chosen=min(candidates,key=lambda x:(x.file_size,x.filename));assert chosen.file_size<60_000_000 and chosen.compress_size<60_000_000
        output=B/Path(chosen.filename).name
        if not output.exists():
            part=output.with_suffix('.mp4.part');total=0;start=time.monotonic()
            with z.open(chosen) as source,part.open('wb') as dest:
                while True:
                    chunk=source.read(512*1024)
                    if not chunk:break
                    dest.write(chunk);total+=len(chunk)
                    if total%(5*1024*1024)==0:print('downloaded',total,flush=True)
            assert total==chosen.file_size;part.replace(output);print('seconds',time.monotonic()-start,flush=True)
        import zlib
        crc=0
        with output.open('rb') as src:
            for chunk in iter(lambda:src.read(1024*1024),b''):crc=zlib.crc32(chunk,crc)
        assert crc==chosen.CRC
        receipt=dict(status='local_public_video_download_verified',pilot_selection='Smallest bottom-camera member, independent of clinical score. One participant pilot cannot establish accuracy.',file=str(output),bytes=output.stat().st_size,sha256=sha(output),zip_member=chosen.filename,zip_member_crc32=crc,crc_verified=True,whole_zip_md5_verified=False,source=f['download_url'],source_page='https://springernature.figshare.com/articles/dataset/Walking_videos/19929893',license=meta['license'],camera='Fixed frontal/back smartphone view at111cm; not original side-following GDI camera. New TUG model feasibility only. Videos and Xsens are NOT synchronized; no framewise 3D target alignment without separate synchronization.',target='Tentative TUG preference pending user reply. Toronto has14participants with actual TUG, no SPPB in described release. Clinical ID join and target-actor/gait-window gates precede score prediction.',external_upload=False)
        (H/'pilot-video.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(receipt['file'],receipt['bytes'],flush=True)
if __name__=='__main__':main()
