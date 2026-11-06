"""Download one stable-order candidate per label to D and check decoding only.

This is an access/format smoke check, NOT gait, target, or frame alignment QA.
"""
import csv
import json
import hashlib
import subprocess
import sys

from audit_gavd import HERE, DATA, write_json
from prepare_comparison import ROOT, read_csv


def main():
    rows=read_csv(HERE/'review_template.csv')
    folder=DATA/'media_samples';folder.mkdir(parents=True,exist_ok=True)
    reports=[]
    for label in ['Normal Gait','Abnormal Gait']:
        r=next(r for r in rows if r['label']==label)
        video=folder/(r['video_id']+'.mp4')
        result=dict(label=label,video_id=r['video_id'],seq=r['record_id'],path=str(video),
                    status='',source_frame_mapping='not_visually_verified',pose_extracted=False)
        if not video.exists():
            cmd=[sys.executable,'-m','yt_dlp','--no-playlist','--no-cache-dir','--no-warnings',
                 '--socket-timeout','20','--retries','0','--max-filesize','100M',
                 '-f','bestvideo[ext=mp4][vcodec^=avc1][height<=720]/best[ext=mp4][height<=720]',
                 '-o',str(video),'--',r['url']]
            try:
                p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',timeout=180)
                (folder/(r['video_id']+'.download.log')).write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
                if p.returncode or not video.exists():
                    result.update(status='download_failed',error=p.stderr[-600:]);reports.append(result);continue
            except subprocess.TimeoutExpired:
                result.update(status='download_timeout');reports.append(result);continue
        code="""import cv2,json,sys
v=cv2.VideoCapture(sys.argv[1])
fps=v.get(cv2.CAP_PROP_FPS);count=int(v.get(cv2.CAP_PROP_FRAME_COUNT))
width=int(v.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
v.set(cv2.CAP_PROP_POS_FRAMES,int(sys.argv[2]));decoded=0
for i in range(124):
    ok,frame=v.read()
    if not ok:break
    decoded+=1
v.release()
print(json.dumps(dict(fps=fps,total_frames=count,width=width,height=height,decoded_frames=decoded)))
"""
        p=subprocess.run([str(ROOT/'tmp/pose-python37/python.exe'),'-c',code,str(video),r['annotation_frame_first']],
                         capture_output=True,text=True,encoding='utf-8',timeout=60)
        if p.returncode:
            result.update(status='decode_error',error=p.stderr[-600:])
        else:
            result.update(json.loads(p.stdout),bytes=video.stat().st_size,
                          sha256=hashlib.sha256(video.read_bytes()).hexdigest())
            result['status']='decoded_124_frames' if result['decoded_frames']==124 else 'short_decode'
        reports.append(result)
        print(json.dumps(result),flush=True)
    write_json(HERE/'media_sample_checks.json',reports)


if __name__=='__main__':main()
