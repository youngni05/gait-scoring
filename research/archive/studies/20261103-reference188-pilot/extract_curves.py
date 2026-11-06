"""Extract source-aligned mean curves; clinical outcomes are not analysed here."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import h5py,numpy as np
DATA=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
VARIABLES=[x.split('.') for x in PROTOCOL['variables']]
SOURCE=json.loads((ROOT/'raw-download-verification.json').read_text())
LINK=json.loads((ROOT/'person-linkage.json').read_text())
audit={};arrays={}
for cohort,fid,sides in [('healthy',42450096,'LR'),('stroke',42450087,'PN')]:
 source=next(x for x in SOURCE if x['file_id']==fid)
 assert source['provider_md5_match']
 rows=LINK[cohort]['rows'];curves=np.full((len(rows),2,9,1001),np.nan);all_var_means=np.full_like(curves,np.nan)
 report=[]
 with h5py.File(source['path'],'r') as f:
  for i,row in enumerate(rows):
   info={'mat_index':i+1,'id':row['supplement_candidate_ids'][0],'sides':{}}
   subchar=f[f['Sub']['sub_char'][i,0]]
   for key,value in row['values'].items():assert np.array_equal(subchar[key][()].ravel(),np.asarray(value).ravel())
   for j,side in enumerate(sides):
    group=f[f['Sub'][f'{side}sideSegm_{side}sideData'][i,0]]
    values=np.stack([group[var][axis][()] for var,axis in VARIABLES],axis=1)
    assert values.ndim==3 and values.shape[1:]==(9,1001),(cohort,i,side,values.shape)
    valid=np.isfinite(values).all(axis=(1,2))
    finite=np.isfinite(values);counts=finite.sum(axis=0)
    np.divide(np.where(finite,values,0).sum(axis=0),counts,out=all_var_means[i,j],where=counts>0)
    if valid.any():curves[i,j]=values[valid].mean(axis=0)
    info['sides'][side]={'strides_total':len(valid),'strides_complete9':int(valid.sum()),'rejected_stride_indices_1based':(np.flatnonzero(~valid)+1).tolist(),'nonfinite_by_variable':{'.'.join(v):int((~finite[:,k,:]).sum()) for k,v in enumerate(VARIABLES)},'finite_minmax':{'.'.join(v):([float(values[:,k,:][finite[:,k,:]].min()),float(values[:,k,:][finite[:,k,:]].max())] if finite[:,k,:].any() else None) for k,v in enumerate(VARIABLES)}}
   info['usable']=bool(np.isfinite(curves[i]).all());report.append(info)
   if (i+1)%20==0:print(cohort,i+1,'/',len(rows),flush=True)
 arrays[cohort]=curves;arrays[cohort+'_allvariable_means']=all_var_means;audit[cohort]=report
 summary={'n':len(report),'usable':sum(r['usable'] for r in report),'strides_total':sum(s['strides_total'] for r in report for s in r['sides'].values()),'complete9_strides':sum(s['strides_complete9'] for r in report for s in r['sides'].values()),'excluded_people':[r['id'] for r in report if not r['usable']]}
 print('SUMMARY',cohort,summary,flush=True);audit[cohort+'_summary']=summary
np.savez_compressed(DATA/'person-curves.npz',**arrays)
(ROOT/'extraction-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
(ROOT/'extraction-provenance.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'array_path':str(DATA/'person-curves.npz'),'array_sha256':hashlib.sha256((DATA/'person-curves.npz').read_bytes()).hexdigest(),'source':SOURCE},indent=2),encoding='utf-8')
