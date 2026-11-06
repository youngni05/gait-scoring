"""Label-blind pause scouting; full-record audit is never a model input."""
from pathlib import Path
import sys,json,hashlib,importlib.util
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'20261031-opencap-speed-turn-refinement'
TURN=HERE.parent/'20261031-opencap-turn'
sys.path.insert(0,str(TURN))
from common import np,pd,read,sha
from scipy.ndimage import uniform_filter1d
spec=importlib.util.spec_from_file_location('prior_scout',PREV/'scout.py')
scout=importlib.util.module_from_spec(spec);spec.loader.exec_module(scout)

def save(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def grid(a,w):
    t=np.arange(round((w[1]-w[0])*30)+1)/30+w[0]
    return t,np.column_stack([np.interp(t,a['time'],a[k]) for k in ['pelvis_tx','pelvis_tz']])

def motion(pos,axis):
    smooth=uniform_filter1d(pos,7,axis=0,mode='nearest')
    v=np.gradient(smooth,1/30,axis=0)
    return np.linalg.norm(v,axis=1),np.abs(v@axis)

def ranges(mask):
    d=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

def overlaps(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0]))

def select(a,m):
    legs=scout.eligible(a,m)[:2]; candidates=[];events=[]
    for leg in legs:
        if leg is None:continue
        t,pos=grid(a,[leg[0],leg[0]+np.floor((leg[1]-leg[0])*30+1e-5)/30]);h,ax=motion(pos,m['axis'])
        for kind,v in [('horizontal',h),('axial',ax)]:
            for l,r in ranges(v[:-1]<.1):
                if (r-l)/30<.5:continue
                w=[float(t[l]),float(t[r])]
                events.append({'kind':kind,'window':w,'seconds':(r-l)/30,
                    'outside_old3_seconds':max(0,(r-l)/30-sum(overlaps(w,b) for b in m.get('tug_walk_windows',[])))})
        for j in range(0,len(t)-45,3):
            w=[float(t[j]),float(t[j]+1.5)]
            # Selection uses axis-progress variation, not target or total delay duration.
            score=float(np.mean(ax[j:j+45]));candidates.append((score,w))
    if not candidates:return {'events':events,'status':'no1.5second_leg'}
    _,candidate=min(candidates,key=lambda x:(x[0],x[1][0]))
    windows=[candidate];cells=[]
    for l,r in m.get('tug_walk_windows',[]):
        for j in range(round((r-l)*30)):
            cell=[l+j/30,l+(j+1)/30]
            if overlaps(cell,candidate)<1e-6:cells.append(cell)
    for left,right in cells[:45]:
        if len(windows)>1 and abs(windows[-1][1]-left)<1e-6:windows[-1][1]=right
        else:windows.append([float(left),float(right)])
    remaining=max(0,45-len(cells))/30
    return {'events':events,'status':'passed' if remaining<1e-6 else 'audit_only','candidate':candidate,'windows':windows,'remaining':remaining}

def main():
    rows=json.loads((HERE.parent/'20261031-opencap-leave283/geometry/manifest.json').read_text())
    common={m['id']:m for m in json.loads((PREV/'common176/manifest.json').read_text())}
    result=[]
    for m in rows:
        m=common.get(m['id'],m);a=read(TURN/'raw'/m['turn_file'])
        assert sha(TURN/'raw'/m['turn_file'])==m['turn_sha256']
        result.append({'id':m['id'],'target':m['target'],'in176':m['id'] in common,**select(a,m)})
    save('audit.json',result)
    index=json.loads((HERE.parent/'20261030-opencap415-tug-timeseries/archive-index.json').read_text())
    from collections import Counter
    save('archive-content-audit.json',dict(Counter(Path(x['name']).suffix for x in index if x['size'])))
    print(json.dumps({'people':len(result),'long':[r for r in result if r['target']>=20],
      'common_missing_pause':sum(any(e['kind']=='horizontal' and e['outside_old3_seconds']>=1 for e in r['events']) for r in result if r['in176'])}),flush=True)
if __name__=='__main__':main()
