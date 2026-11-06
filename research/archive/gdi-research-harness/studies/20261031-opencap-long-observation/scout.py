from pathlib import Path
import sys,json,hashlib,importlib.util
HERE=Path(__file__).resolve().parent
DELAY=HERE.parent/'20261031-opencap-delay'
PREV=HERE.parent/'20261031-opencap-speed-turn-refinement'
TURN=HERE.parent/'20261031-opencap-turn'
sys.path.insert(0,str(TURN))
from common import np,pd,read,sha
spec=importlib.util.spec_from_file_location('prior_scout',PREV/'scout.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def save(n,v):
    p=HERE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    rows=json.loads((PREV/'common176/manifest.json').read_text());result=[]
    for m in rows:
        a=read(TURN/'raw'/m['turn_file']);pre,post,extent=old.eligible(a,m)
        start=pre[0] if pre else m['tug_walk_windows'][0][0]
        pos=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);v=np.linalg.norm(np.diff(pos,axis=0),axis=1)/np.diff(a['time'])
        result.append({'id':m['id'],'target':m['target'],'source_start':float(a['time'][0]),'source_end':float(a['time'][-1]),'start':start,'available_seconds':float(a['time'][-1]-start),'pre':pre,'post':post,'extent':extent,'peak_all_after_start':float(max(v[a['time'][:-1]>=start]))})
    save('scout.json',result)
    print(json.dumps({'people':len(result),'available_quantiles':np.quantile([r['available_seconds'] for r in result],[0,.25,.5,.75,1]).tolist(),
      'exact_budget_counts':{str(b):{'all':sum(r['available_seconds']>=b for r in result),'long':sum(r['available_seconds']>=b and r['target']>=20 for r in result)} for b in [10,20,30]},
      'new_jump_people':[r['id'] for r in result if r['peak_all_after_start']>6], 'long':[r for r in result if r['target']>=20]}),flush=True)
if __name__=='__main__':main()
