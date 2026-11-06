"""Input-only diagnostic: terminal tracking drop can shift a global height threshold."""
from audit import *
def main():
    rows=[]
    for s in json.loads((HERE/'marker-boundaries.json').read_text()):
        if 'tug_cone' not in s['file']:continue
        fps,t,names,xyz=trc(HERE/s['file']);h=(xyz[:,list(names).index('RHJC_study'),1]+xyz[:,list(names).index('LHJC_study'),1])/2
        for remove in [0,1,round(.1*fps),round(.5*fps)]:
            end=len(t)-remove;hh=h[:end];b=author_bounds(fps,t[:end],names,xyz[:end].copy())
            rows.append({'id':s['id'],'removed_tail_frames':remove,'removed_tail_seconds':remove/fps,'minimum_height':float(hh.min()),'maximum_height':float(hh.max()),'absolute30percent_threshold':float(hh.min()+.3*(hh.max()-hh.min())),'last_frame_height':float(hh[-1]),'maximum_last_halfsecond_vertical_speed':float(max(abs(np.diff(hh[-round(.5*fps):])))*fps),'boundary':b})
    save('marker-tail-sensitivity.json',rows);print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__':main()
