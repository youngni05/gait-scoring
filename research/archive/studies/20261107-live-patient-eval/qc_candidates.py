"""Required-coordinate missing fraction and track continuity for each candidate. No GDI model runs here."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'tmp/ljmu-analysis-python-deps'),str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle')]
import numpy as np
from preprocess import extract_windows
POSE=Path('D:/gait-research-data/D_P/live-patient-eval-20261107/body25')
rows=json.loads((HERE/'candidates.json').read_text(encoding='utf-8'))
out=[]
for row in rows:
    for ci,c in enumerate(row['candidates']):
        if c['moving_fraction']<.8:continue
        stem=row['record'];raw=np.full((124,75),np.nan)
        for k,sel in enumerate(c['person_indices']):
            if sel>=0:
                p=json.loads((POSE/stem/('%s_%012d_keypoints.json'%(stem,c['start']+k))).read_text())['people'];raw[k]=p[sel]['pose_keypoints_2d']
        a=raw.reshape(124,25,3);v=np.isfinite(a[:,8,0])&(a[:,8,2]>.1);i=np.flatnonzero(v)
        hip=a[i,8,0];step=np.abs(np.diff(hip))/np.maximum(np.diff(i),1)
        thigh=np.linalg.norm(a[:,[9,12],:2]-a[:,[10,13],:2],axis=2);t=np.nanmedian(thigh[(a[:,[9,12],2]>.1)&(a[:,[10,13],2]>.1)])
        d=dict(record=stem,group=row['group'],person=row['person'],candidate=ci,start=c['start'],moving=c['moving_fraction'],observed=c['observed_fraction'],thigh=float(t),direction=float(np.sign(hip[-1]-hip[0])),hip_span=float(np.ptp(hip)),max_hip_step_over_thigh=float(step.max()/t),reversals=int((np.diff(np.sign(np.diff(np.convolve(hip,np.ones(9)/9,'valid'))))!=0).sum()))
        try:
            e=extract_windows(raw);d.update(passes=e['window_start'].tolist()==[0],missing=float(e['window_missing_fraction'][0]))
        except ValueError as err:d.update(passes=False,reason=str(err))
        out.append(d)
(HERE/'candidates-qc.json').write_text(json.dumps(out,indent=1)+'\n',encoding='utf-8')
good=[d for d in out if d['passes']];print(len(out),len(good),len({d['record'] for d in good}),len({d['person'] for d in good}))
for d in out:print(d['record'][:18],d['candidate'],d['group'],round(d['moving'],2),round(d.get('missing',9),3),d['passes'],'step',round(d['max_hip_step_over_thigh'],2),'span',round(d['hip_span']),'th',round(d['thigh']))
