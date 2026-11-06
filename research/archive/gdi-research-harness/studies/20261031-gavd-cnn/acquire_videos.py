"""Acquire selected public source videos at annotation resolution, without resampling."""
import argparse,ast,concurrent.futures,csv,hashlib,json,subprocess,sys,time
from pathlib import Path
from audit_gavd import HERE,DATA,write_json
from prepare_comparison import read_csv,ROOT

def acquire(row):
    folder=DATA/'videos';folder.mkdir(parents=True,exist_ok=True)
    video=folder/(row['video_id']+'.mp4')
    height=json.loads(row['annotation_video_info'])['height']
    result=dict(video_id=row['video_id'],path=str(video),annotation_height=height,status='')
    if not video.exists():
        cmd=[sys.executable,'-m','yt_dlp','--no-playlist','--no-cache-dir','--no-warnings',
             '--socket-timeout','20','--retries','1','--max-filesize','500M',
             '-f','bestvideo[ext=mp4][vcodec^=avc1][height<=%s]/best[ext=mp4][height<=%s]'%(height,height),
             '-o',str(video),'--',row['url']]
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',timeout=300)
            (folder/(row['video_id']+'.download.log')).write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
            if p.returncode or not video.exists():return dict(result,status='download_failed',error=p.stderr[-800:])
        except subprocess.TimeoutExpired:return dict(result,status='timeout')
    code="""import cv2,json,sys
v=cv2.VideoCapture(sys.argv[1]);ok,im=v.read()
print(json.dumps(dict(opened=bool(ok),fps=v.get(cv2.CAP_PROP_FPS),frames=int(v.get(cv2.CAP_PROP_FRAME_COUNT)),width=int(v.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
v.release()
"""
    p=subprocess.run([str(ROOT/'tmp/pose-python37/python.exe'),'-c',code,str(video)],capture_output=True,text=True,encoding='utf-8',timeout=60)
    if p.returncode:return dict(result,status='decode_failed',error=p.stderr[-800:])
    return dict(result,status='decoded_metadata',**json.loads(p.stdout),bytes=video.stat().st_size,
                sha256=hashlib.sha256(video.read_bytes()).hexdigest())

def main(args):
    rows=read_csv(args.manifest or HERE/'review_template.csv')
    if args.label:rows=[r for r in rows if r['label']==args.label]
    videos={r['video_id']:r for r in rows}
    selected={r['record_id'] for r in rows}
    boxdir=DATA/'annotations_by_sequence';boxdir.mkdir(parents=True,exist_ok=True)
    boxrows={s:[] for s in selected}
    source=json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8'))
    for info in source['files']:
        with Path(info['local_path']).open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if r['seq'] in selected:boxrows[r['seq']].append(dict(frame=int(r['frame_num']),bbox=ast.literal_eval(r['bbox'])))
    for seq,boxes in boxrows.items():write_json(boxdir/(seq+'.json'),sorted(boxes,key=lambda r:r['frame']))
    cache_path=DATA/'video_inventory.json'
    cache=json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    pending=[r for k,r in sorted(videos.items()) if cache.get(k,{}).get('status')!='decoded_metadata']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(acquire,pending):
            cache[result['video_id']]=result;write_json(cache_path,cache)
            print(json.dumps(result),flush=True)
    write_json(HERE/(args.summary+'.json'),{k:cache[k] for k in sorted(videos)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label');p.add_argument('--manifest');p.add_argument('--summary',default='normal_video_metadata')
    main(p.parse_args())
