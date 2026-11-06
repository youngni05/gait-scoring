"""Input-only scout; never selects windows using clinical TUG labels."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent
TURN=HERE.parent/'20261031-opencap-turn'
PRIOR=HERE.parent/'20261031-opencap-leave283/geometry'
sys.path.insert(0,str(TURN))
from common import np,pd,read,sha
from scipy.ndimage import uniform_filter1d
import geometry

def save(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def runs(mask,t):
    delta=np.diff(np.r_[False,mask,False].astype(int))
    return [(float(t[a]),float(t[b])) for a,b in zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1)-1) if t[b]-t[a]>=.25]

def eligible(a,m):
    t=a['time'];pos=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);dt=np.median(np.diff(t))
    smooth=uniform_filter1d(pos,max(3,round(.25/dt)),axis=0,mode='nearest')
    origin=np.median(smooth[:max(1,round(.5/dt))],axis=0);progress=(smooth-origin)@m['axis']
    extent=float(np.quantile(progress,.99));away=progress>=max(.3,.1*extent)
    pre=runs(away&(t<=m['turn_start']+1e-7),t)
    post=runs(away&(t>=m['turn_end']-1e-7),t)
    # Keep the leg immediately adjacent to the pre-existing turn; no search for fastest walking.
    pre=[w for w in pre if abs(w[1]-m['turn_start'])<.05]
    post=[w for w in post if abs(w[0]-m['turn_end'])<.05]
    return pre[-1] if pre else None,post[0] if post else None,extent

def choose(a,m):
    pre,post,extent=eligible(a,m)
    # Native recording frequencies vary; align within endpoints on a 30Hz duration lattice.
    caps=[max(0,int(np.floor((w[1]-w[0])*30+1e-5))) if w else 0 for w in [pre,post]]
    if sum(caps)<90:return {'status':'less_than3seconds_nonturn_walk','pre':pre,'post':post,'caps':caps,'extent':extent}
    npre=min(caps[0],45);npost=min(caps[1],90-npre)
    if npre+npost<90:npre=min(caps[0],90-npost)
    assert npre+npost==90
    windows=[]
    if npre: windows.append([pre[1]-npre/30,pre[1]])
    if npost:windows.append([post[0],post[0]+npost/30])
    peak=max(geometry.root_qc(a,*w) for w in windows)
    return {'status':'passed' if peak<=6 else 'horizontal_tracking_jump','pre':pre,'post':post,'caps':caps,'extent':extent,'windows':windows,'max_horizontal_speed':peak}

def main():
    rows=json.loads((PRIOR/'manifest.json').read_text());audit=[]
    for m in rows:
        path=TURN/'raw'/m['turn_file'];assert sha(path)==m['turn_sha256'];a=read(path)
        r=choose(a,m);audit.append({'id':m['id'],'target':m['target'],**r})
    save('scout.json',audit)
    from collections import Counter
    print(json.dumps({'counts':dict(Counter(r['status'] for r in audit)),'failures':[r for r in audit if r['status']!='passed'],'long':[r for r in audit if r['target']>=20]}),flush=True)
if __name__=='__main__':main()
