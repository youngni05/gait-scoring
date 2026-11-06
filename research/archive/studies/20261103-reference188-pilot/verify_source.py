"""Replay source extraction against the provider's separate sagittal summary."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np,h5py,openpyxl
DATA=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot')
OLD=Path('D:/gait-research-data/D_P/dataset-audits/20261102-sideview-reference')
source=json.loads((ROOT/'raw-download-verification.json').read_text())
archive=np.load(DATA/'person-curves.npz');arrays={key:archive[key] for key in archive.files};protocol=json.loads((ROOT/'protocol.json').read_text())
variables=['AnkleAngles','KneeAngles','HipAngles','PelvisAngles'];out={}
for cohort,fid,sides,xlsx in [('healthy',42450096,'LR','MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx'),('stroke',42450087,'PN','MAT_normalizedData_PostStrokeAdults_v27-02-23.xlsx')]:
 w=openpyxl.load_workbook(OLD/xlsx,read_only=True,data_only=True);rows=[]
 with h5py.File(next(s['path'] for s in source if s['file_id']==fid),'r') as f:
  for i in range(len(arrays[cohort])):
   sheet=w[f'Sub{i+1:02d}'];it=sheet.values;header=next(it);table=list(it)
   target={name:np.array([r[j] for r in table],dtype=float) for j,name in enumerate(header) if (name in variables or any(name==side+'side_'+v for side in 'PN' for v in variables))}
   errors={}
   for v in variables:
    raw=[f[f['Sub'][f'{side}sideSegm_{side}sideData'][i,0]][v]['x'][()] for side in sides]
    rawmeans=[np.nanmean(a,axis=0) for a in raw]
    k=protocol['variables'].index(v+'.x')
    for j in range(2):assert np.allclose(rawmeans[j],arrays[cohort+'_allvariable_means'][i,j,k],rtol=0,atol=1e-10)
    if cohort=='healthy':
     equal=np.mean(rawmeans,axis=0);pooled=np.nanmean(np.concatenate(raw),axis=0)
     errors[v]={'equal_side_max_abs':float(np.max(np.abs(equal-target[v]))),'pooled_stride_max_abs':float(np.max(np.abs(pooled-target[v])))}
    else:
     errors[v]={side+'_max_abs':float(np.max(np.abs(rawmeans[j]-target[side+'side_'+v]))) for j,side in enumerate(sides)}
   rows.append({'mat_index':i+1,'errors_degrees':errors})
 out[cohort]=rows
 print(cohort,'checked',len(rows),'people',flush=True)
(ROOT/'published-summary-check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Source comparison saved. Inspect errors before making verification claims.')
