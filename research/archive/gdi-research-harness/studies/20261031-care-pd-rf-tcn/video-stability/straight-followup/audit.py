"""Geometry-only audit; never loads scores, features or RF models."""
import json, hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=Path('D:/gait-research-data/MoVi/experiments/walking-expansion')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x): (HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def geometry(r):
    angles=[]
    for a,b in [(6,7),(0,1)]:
        axis=r[:,b,:2]-r[:,a,:2]
        yaw=np.arctan2(axis[:,1],axis[:,0]); valid=np.isfinite(yaw)
        yaw[valid]=np.unwrap(yaw[valid]);angles.append(np.rad2deg(yaw))
    return angles,(r[:,6,:2]+r[:,7,:2])/2
def metrics(angles,pelvis,k,n):
    a=np.array([x[k:k+n] for x in angles]); p=pelvis[k:k+n]
    displacement=float(np.linalg.norm(p[-1]-p[0]))
    path=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    return dict(heading_p90_span=float(np.max(np.percentile(a,95,axis=1)-np.percentile(a,5,axis=1))),
        heading_full_span=float(np.max(np.ptp(a,axis=1))),efficiency=displacement/max(path,1e-9),speed_mps=displacement/1000/((n-1)/30))
def main():
    rows=json.loads((SOURCE/'results.json').read_text())['results']; people=[]; fingerprints={}
    for row in sorted([x for x in rows if x['camera']=='PG1' and x['variant']=='heavy' and x['input_width']==800],key=lambda x:x['subject']):
        sid=row['subject']; zs=[]; masks=[]
        for cam in ['PG1','PG2']:
            path=SOURCE/('subject-%02d'%sid)/'heavy'/(cam+'_800_comparison.npz')
            fingerprints[str(path)]=sha(path); z=np.load(path);zs.append(z)
            xy=z['reference_xy']; masks.append(((xy[:,:,0]>=0)&(xy[:,:,0]<800)&(xy[:,:,1]>=0)&(xy[:,:,1]<600)&~z['reference_gap']).all(1))
        assert np.array_equal(zs[0]['frames'],zs[1]['frames'])
        assert np.array_equal(zs[0]['reference_3d'],zs[1]['reference_3d'],equal_nan=True)
        both=masks[0]&masks[1];angles,pelvis=geometry(zs[0]['reference_3d']); windows={}
        for seconds in [1,1.5,2,2.5,3,4,5]:
            n=round(seconds*30); candidates=[]
            for k in range(len(both)-n+1):
                if both[k:k+n].all():
                    m=metrics(angles,pelvis,k,n);m.update(start=int(zs[0]['frames'][k]),frames=n);candidates.append(m)
            windows[str(seconds)]={}
            for threshold in [10,20,30]:
                eligible=[x for x in candidates if x['heading_p90_span']<=threshold and x['heading_full_span']<=1.5*threshold and x['efficiency']>=.98 and x['speed_mps']>=.2]
                windows[str(seconds)][str(threshold)]=dict(count=len(eligible),earliest=eligible[0] if eligible else None)
            windows[str(seconds)]['unrestricted']=candidates[0] if candidates else None
            windows[str(seconds)]['max_turn']=max(candidates,key=lambda x:x['heading_p90_span']) if candidates else None
        people.append(dict(subject=sid,available_frames=len(both),both_visible_frames=int(both.sum()),windows=windows))
    summary=[]
    for sec in [1,1.5,2,2.5,3,4,5]:
        summary.append(dict(seconds=sec,visible=sum(x['windows'][str(sec)]['unrestricted'] is not None for x in people),
            **{str(t):[x['subject'] for x in people if x['windows'][str(sec)][str(t)]['count']>0] for t in [10,20,30]}))
    result=dict(criteria='Both cameras all 12 reference joints in frame and no gap. Max of hip and shoulder p95-p05 yaw span <= threshold; full span <= 1.5 threshold; pelvis XY displacement/path >= .98; net speed >= .2 m/s. Thresholds are geometric sensitivity settings, not clinical standards. No RF outputs read.',
        primary_heading_degrees=20,people=people,availability=summary,source_sha256=fingerprints,script_sha256=sha(Path(__file__)))
    save('availability.json',result)
    print(json.dumps([{**{k:v for k,v in r.items() if k not in ['10','20','30']},**{k:len(r[k]) for k in ['10','20','30']}} for r in summary]))
if __name__=='__main__':main()
