from context import *
sys.path.append(str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 rows=json.loads((PRIOR/'geometry/manifest.json').read_text());audit=[];plots=[]
 for m in rows:
  a=read(PRIOR/'raw'/m['turn_file']);r={'id':m['id'],'target':m['target']}
  try:
   c=chair(a,m);r.update(c);r['status']='chair_found';extra=sum(c[k][1]-c[k][0] for k in ['rise','sit']);r['total_observation']=3+m['turn_observed_seconds']+extra
   try:r['central']=central(a,m,m['turn_observed_seconds']+extra,[c['rise'][1],c['sit'][0]]);r['matched']='yes'
   except ValueError as e:r['matched']=str(e)
  except ValueError as e:r['status']=str(e)
  audit.append(r)
  if m['target']>=20:plots.append((m,a,r))
 save('scout.json',audit)
 for page in range(3):
  fig,axes=plt.subplots(3,1,figsize=(13,10))
  for ax,(m,a,r) in zip(axes,plots[page*3:page*3+3]):
   t=a['time'];ax.plot(t,a['pelvis_ty'],label='pelvis height');ax.axvspan(m['turn_start'],m['turn_end'],alpha=.2,color='orange',label='turn')
   for key,color in [('rise','green'),('sit','purple')]:
    if key in r:ax.axvspan(*r[key],alpha=.2,color=color,label=key)
   ax.set_title(m['id']+' / '+r['status']+' / '+r.get('matched',''));ax.set_ylabel('height (m)');ax.set_xlabel('record time (s)');ax.legend(loc='upper right');ax.grid(alpha=.2)
  fig.tight_layout();fig.savefig(STUDY/f'chair-review-{page+1}.png',dpi=120);plt.close(fig)
 print(pd.DataFrame(audit).groupby('status').size().to_string());print(pd.DataFrame(audit).groupby('matched').size().to_string());print(json.dumps([r for r in audit if r['target']>=20]),flush=True)
if __name__=='__main__':main()
