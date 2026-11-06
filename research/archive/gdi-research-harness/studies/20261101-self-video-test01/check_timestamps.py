"""Read actual MP4 presentation timestamps; do not alter or resample frames."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'D:/gait-research-data/D_P/video-caption-jev-20261031/deps')
import av
def main():
    out=[]
    for r in json.loads((HERE/'inventory.json').read_text(encoding='utf8')):
        with av.open(r['file']) as c:
            stream=c.streams.video[0];times=[float(f.pts*f.time_base) for f in c.decode(stream)]
        assert len(times)==r['frames'];steps=[b-a for a,b in zip(times,times[1:])];expected=1/r['fps']
        unusual=[dict(after_frame=i,seconds=v) for i,v in enumerate(steps) if abs(v-expected)>.001]
        item=dict(number=r['number'],id=r['id'],decoded_frames=len(times),first_pts=times[0],last_pts=times[-1],minimum_step=min(steps),maximum_step=max(steps),unusual_intervals=unusual,monotonic=all(v>0 for v in steps),frame_pts_seconds=times)
        out.append(item);print(r['id'],'frames',len(times),'irregular_intervals',len(unusual),flush=True)
        (HERE/'timestamps.json').write_text(json.dumps(out,indent=2),encoding='utf8')
if __name__=='__main__':main()
