"""Read-only original XLSX/CSV audit; no new fitting or outcome-based selection."""
from pathlib import Path
import json,csv,zipfile,xml.etree.ElementTree as ET,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104')
G=Path('D:/gait-research-data/D_P/planar-tug-improvement-20261104/gstride')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ranks(a):
    _,inv,count=np.unique(a,return_inverse=True,return_counts=True)
    return (np.cumsum(count)-.5*(count-1))[inv]
def rho(a,b):return float(np.corrcoef(ranks(a),ranks(b))[0,1])
def stat(a):return dict(n=len(a),mean=float(np.mean(a)),min=float(np.min(a)),p25=float(np.quantile(a,.25)),median=float(np.median(a)),p75=float(np.quantile(a,.75)),max=float(np.max(a)))
def main():
    ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};src=G/'GSTRIDE_DDBB.xlsx'
    with zipfile.ZipFile(src) as z:
      strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))];rows=[]
      wb=ET.fromstring(z.read('xl/workbook.xml'));sheet=wb.find('x:sheets',ns)[0].get('name')
      for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//x:sheetData/x:row',ns):
        r={}
        for c in row:
          v=c.find('x:v',ns)
          if v is None:continue
          val=strings[int(v.text)] if c.get('t')=='s' else v.text
          if c.get('t') not in ['s','str']:
            try:val=float(val)
            except ValueError:pass
          r[c.get('r')]=val
        if r:rows.append(r)
    assert rows==next(iter(json.loads((G/'workbook-cells.json').read_text()).values()))
    def val(r,col):return next((v for k,v in r.items() if ''.join(x for x in k if x.isalpha())==col),None)
    raw={val(r,'B'):r for r in rows[3:]};d=np.load(BIG/'input.npz');p=np.load(BIG/'verified-predictions.npz');gm=d['cohort']=='G';fm=d['cohort']=='F';ids=d['ids'][gm];y=d['y'][gm]
    assert len(ids)==142 and np.array_equal(d['ids'],p['ids'])
    assert all(val(raw[pid],'AD')==target for pid,target in zip(ids,y))
    labels={}
    for col in ['Q','AD']:
        a=np.array([val(raw[pid],col) for pid in ids]);labels[col]={'header':val(rows[1],col),'unit':'seconds','range':f'{col}4:{col}166',**stat(a)}
    pred=p['G_only__new__core3'][gm];error=abs(pred-y);baseline=np.zeros(len(y))
    for f in range(5):baseline[d['fold'][gm]==f]=np.median(y[d['fold'][gm]!=f])
    bands=[]
    for lo,hi in [(0,8),(8,12),(12,20),(20,999)]:
        m=(y>=lo)&(y<hi);bands.append(dict(lower_inclusive=lo,upper_exclusive=hi,n=int(m.sum()),mae=float(error[m].mean()),contribution_to_total_mae=float(error[m].sum()/len(y))))
    features={};whole={}
    for cohort,mask in [('FRAILPOL',fm),('GSTRIDE',gm)]:
      features[cohort]={}
      for j,name in enumerate(['speed','cycle','swing']):
        a=d['new'][mask,j];features[cohort][name]={**stat(a),'rho_TUG':rho(a,d['y'][mask])}
    for col,j,div,name in [('AP',0,1,'speed'),('AJ',1,1,'cycle'),('AQ',2,100,'swing')]:
        a=np.array([val(raw[pid],col) for pid in ids])/div
        whole[name]={**stat(a),'rho_TUG':rho(a,y),'rho_selected_short':rho(a,d['new'][gm,j]),'scope':'provider whole-record summaries; diagnostic only, not a deployable short-video input'}
    oldaudit=json.loads((HERE.parent/'20261104-planar-tug-improvement/gstride-short-window-audit.json').read_text());selected={r['id']:r['stride_indices'][0] for r in oldaudit['included']};examples=[];source_hashes={};medians=[]
    for i,pid in enumerate(ids):
        path=G/'processed/_GAIT/_PARAMETERS'/f'{pid}_GAIT_PARAMETERS.csv';rr=list(csv.DictReader(path.open()));source_hashes[str(path)]=sha(path)
        valid=[]
        for q in rr:
          dt=float(q['CycleDuration[s]']);sp=float(q['Stride_Velocity[m/s]']);length=float(q['StrideLength[m]']);sw=float(q['Per_Swing[%CD]']);turn=float(q['TurningAngle[degrees]'])
          if np.isfinite([dt,sp,length,sw,turn]).all() and .6<=dt<=3 and .05<=sp<=2.5 and .1<=length<=3 and 0<=sw<=100 and abs(turn)<20:valid.append(sp)
        med=float(np.median(valid));medians.append(med)
        if pid in ['V001','V026','V064','V124','V138']:
          examples.append(dict(id=str(pid),TUG=float(y[i]),selected_stride_index=selected[pid],selected_speed=float(d['new'][gm,0][i]),record_valid_stride_speed_median=med,record_valid_strides=len(valid),provider_record_mean_speed=val(raw[pid],'AP'),clinical_4m_speed=val(raw[pid],'R'),prediction=float(pred[i]),absolute_error=float(error[i])))
    medians=np.array(medians);short=d['new'][gm,0]
    out=dict(status='audit_complete_no_new_fits',workbook_sha256=sha(src),sheet=sheet,source_row_link_verified=142,labels=labels,
      TUG_under7=int(sum(y<7)),TUG20plus=int(sum(y>=20)),prediction=stat(pred),MAE=float(error.mean()),median_absolute_error=float(np.median(error)),
      train_only_median_baseline_MAE=float(abs(baseline-y).mean()),error_bands=bands,feature_distributions=features,provider_record_diagnostics=whole,
      selection_audit=dict(first_stride_selected=int(sum(selected[pid]==1 for pid in ids)),selected_speed_under_half_record_valid_stride_median=int(sum(short<.5*medians)),record_valid_stride_median_distribution=stat(medians)),
      examples=examples,source_parameter_hashes=source_hashes,
      interpretation='Arithmetic/ID/target column correct. First-eligible-stride selection can represent a much slower movement than the remaining recording; start/transition explanation is plausible but not event-labelled. Current high MAE must not be attributed solely to model transfer or intrinsic absence of useful gait information.',
      scope='Whole-record/clinical4m metrics used only for diagnosis, not added to mobile input or model training. No new model fit or score-dependent window selection.')
    dest=HERE/'gstride-label-input-audit.json';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k!='source_parameter_hashes'},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
