"""Recount eligible people without using audit's window implementation."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
    saved=json.loads((HERE/'availability.json').read_text());root=Path('D:/gait-research-data/MoVi/experiments/walking-expansion');counts={};maxerror=0.
    for duration in [1,1.5,2,2.5,3,4,5]:
        ids={str(t):[] for t in [10,20,30]};visible=[];length=round(duration*30)
        for person in saved['people']:
            sid=person['subject'];paired=[np.load(root/('subject-%02d'%sid)/'heavy'/(cam+'_800_comparison.npz')) for cam in ['PG1','PG2']];good=[]
            for z in paired:
                a=z['reference_xy'];good.append((a[...,0]>=0)&(a[...,0]<800)&(a[...,1]>=0)&(a[...,1]<600)&(~z['reference_gap']))
            ok=(good[0]&good[1]).all(axis=1);r=paired[0]['reference_3d'];found={str(t):False for t in [10,20,30]};seen=False
            for start in range(len(ok)-length+1):
                if not np.all(ok[start:start+length]):continue
                seen=True;window=r[start:start+length];spans=[];ranges=[]
                for a,b in [(6,7),(0,1)]:
                    xy=window[:,b,:2]-window[:,a,:2];angle=np.unwrap(np.arctan2(xy[:,1],xy[:,0]))*180/np.pi;spans.append(np.quantile(angle,.95)-np.quantile(angle,.05));ranges.append(angle.max()-angle.min())
                center=window[:,[6,7],:2].mean(axis=1);dist=np.sqrt(np.square(center[-1]-center[0]).sum());path=np.sqrt(np.square(np.diff(center,axis=0)).sum(axis=1)).sum()
                for t in [10,20,30]:
                    if max(spans)<=t and max(ranges)<=1.5*t and dist/max(path,1e-9)>=.98 and dist/1000/((length-1)/30)>=.2:found[str(t)]=True
            if seen:visible.append(sid)
            for t,yes in found.items():
                if yes:ids[t].append(sid)
        row=next(r for r in saved['availability'] if r['seconds']==duration)
        assert len(visible)==row['visible']
        for t in ids:assert ids[t]==row[t]
        counts[str(duration)]={t:len(v) for t,v in ids.items()}
    result=dict(passed=True,people=30,counts=counts,criterion_unchanged=True)
    (HERE/'availability-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
