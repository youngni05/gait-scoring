import json,sys,numpy as np
from pathlib import Path
sys.path.insert(0,'D:/Projects/D_P/studies/20261103-reference188-pilot')
from inspect_mat import Remote,ROOT,h5py
previous=json.loads(Path('D:/Projects/D_P/studies/20261102-sideview-reference-audit/188-demographics-audit.json').read_text(encoding='utf-8-sig'))
result={}
for name,fid,size in [('healthy',42450096,6244593847),('stroke',42450087,2906058754)]:
 remote=Remote(fid,size);rows=[]
 with h5py.File(remote,'r') as f:
  refs=f['Sub']['sub_char'][()].ravel()
  for i,ref in enumerate(refs,1):
   group=f[ref];row={'mat_index':i,'source_file_id':fid,'values':{}}
   for key in group:
    a=group[key][()]
    row['values'][key]=float(a.ravel()[0]) if a.size==1 else a.tolist()
   v=row['values'];candidates=[]
   for old in previous[name]:
    matches=old['age']==v['Age'] and (old['sex']=='M')==bool(v['Male'])
    if name=='stroke':matches=matches and all(old[k]==v[k] for k in ['FAC','POMA','TIS'])
    else:matches=matches and old['height']==v['Height'] and old['leglength']==v['LegLength'] and abs(old['weight']-v['Weight'])<=0.51
    if matches:candidates.append(old['id'])
   row['supplement_candidate_ids']=candidates;rows.append(row)
   if i%20==0:print(name,i,'of',len(refs),'new blocks',remote.calls,flush=True)
   (ROOT/('metadata-'+name+'-partial.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
 result[name]={'n':len(rows),'age65plus':sum(x['values']['Age']>=65 for x in rows),'unique_matches':sum(len(x['supplement_candidate_ids'])==1 for x in rows),'rows':rows}
 print(name,{k:v for k,v in result[name].items() if k!='rows'},flush=True)
(ROOT/'person-linkage.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
