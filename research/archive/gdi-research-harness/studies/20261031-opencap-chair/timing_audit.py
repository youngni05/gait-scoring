"""Descriptive label-timing audit. None of these elapsed times enter a model."""
from context import *
sys.path.append(str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

audit=json.loads((STUDY/'transition-audit.json').read_text());rows=[]
for r in audit:
 if all('window' in r[k] for k in ['rise','sit']):
  elapsed=r['sit']['window'][1]-r['rise']['window'][0]
  rows.append({'id':r['id'],'clinical_TUG':r['target'],'selected_rise_to_sit_span':elapsed,'clinical_minus_span':r['target']-elapsed})
z=pd.DataFrame(rows);z.to_csv(STUDY/'timing-audit.csv',index=False);save('timing-audit.json',{'people':len(z),'median_clinical_minus_span':float(z.clinical_minus_span.median()),'median_absolute_difference':float(z.clinical_minus_span.abs().median()),'difference_over5s':z[z.clinical_minus_span.abs()>5].to_dict('records'),'long':z[z.clinical_TUG>=20].to_dict('records'),'meaning':'rise/sit core boundaries are not clinically annotated timer start/stop; differences can reflect preparation, trial mismatch or tracking/detection errors; not a substitute ground truth and never model inputs'})
man={r['id']:r for r in json.loads((PRIOR/'geometry/manifest.json').read_text())};fig,axes=plt.subplots(3,1,figsize=(13,9))
for ax,pid in zip(axes,['sub-283','sub-246','sub-310']):
 m=man[pid];a=read(PRIOR/'raw'/m['turn_file']);t=a['time'];r=next(r for r in audit if r['id']==pid)
 for k in ['pelvis_tilt','lumbar_extension','hip_flexion_r','hip_flexion_l']:ax.plot(t,a[k],label=k,linewidth=.8)
 for k,c in [('rise','green'),('sit','purple')]:ax.axvspan(*r[k]['window'],alpha=.15,color=c)
 ax.set_title(pid);ax.set_xlabel('record time (s)');ax.set_ylabel('angle (deg)');ax.legend(fontsize=8,ncol=4);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(STUDY/'preparation-motion-review.png',dpi=110);plt.close(fig)
print(json.dumps({'people':len(z),'median_gap':float(z.clinical_minus_span.median()),'median_abs_gap':float(z.clinical_minus_span.abs().median()),'long':z[z.clinical_TUG>=20].to_dict('records')}),flush=True)
