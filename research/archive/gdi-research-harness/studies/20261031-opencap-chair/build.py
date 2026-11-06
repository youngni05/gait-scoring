from context import *
from segment import features

def fixed(a,event):
 center=round(event['peak']*30)/30;lo=np.ceil(a['time'][0]*30-1e-6)/30;hi=np.floor((a['time'][-1]-2)*30+1e-6)/30;left=float(max(lo,min(center-1,hi)))
 if left+2>a['time'][-1]+1e-6:raise ValueError('fixed_clip_edge')
 return [left,left+2]

def peak_vertical(a,w):
 p=a[(a['time']>=w[0]-1e-7)&(a['time']<=w[1]+1e-7)]
 return float(max(abs(np.diff(p['pelvis_ty'])/np.diff(p['time']))))

def main():
 assert not list(STUDY.glob('*/completion.json'))
 if (STUDY/'preparation.json').exists() and not (STUDY/'preparation-initial.json').exists():save('preparation-initial.json',json.loads((STUDY/'preparation.json').read_text()))
 configs={'fixed_strict':('fixed',False,False),'fixed_flagged':('fixed',True,False),'event_strict':('event',False,False),'rise_strict':('fixed',False,True)}
 protocol={'target':'unchanged same-person same-visit tft_tug seconds; exact timed trial unresolved','population':'same selected TUG and walking files of previous238; no trial substitution','phase_detection':'local positive/negative smoothed pelvis-height velocity peaks near chair; net change>=0.08m; 0.25s margins; sitting endpoint within0.12m of initial height; no manual labels','fixed':'2 seconds around peak, boundary shift only if source supports full2s; actual full transition duration not a model input','event':'local transition core plus0.25s margins; includes duration features','quality':'native vertical speed<=3m/s in selected chair clips; previous horizontal<=6m/s in turn and added context; flagged sensitivity retains vertical failures; no imputed missing phases','observation':'same original walking3s and complete turn D; augmented adds rise/sit clips, control adds exactly same total seconds of non-chair TUG walking around original turn; windows disjoint; no padding or elapsed wholeTUG','views_both':['TurnBase','TurnContext','ChairTime','ChairMotion','ChairAll'],'views_rise':['TurnBase','TurnContext','RiseMotion'],'feature_note':'TurnContext retains original13features and per-side vertical summaries excluding explicit duration, plus horizontal context summaries; ChairMotion omits explicit phase durations; other motion summaries can still encode timing','training':'reuse same frozen outer and inner person folds restricted to each common cohort; RF128 leaf2/5 innerMAE; 5partitions x3seeds; localCPU','metrics':'MAE, bias, positive underprediction; paired person bootstrap conditional on fits; 20s descriptive threshold','configs':configs,'source_code':sha(STUDY/'segment.py'),'source_manifest':sha(PRIOR/'geometry/manifest.json')}
 save('protocol.json',protocol)
 manifests=json.loads((PRIOR/'geometry/manifest.json').read_text());audits={r['id']:r for r in json.loads((STUDY/'transition-audit.json').read_text())};olddata=dict(np.load(PRIOR/'geometry/input.npz'));data={m['id']:read(PRIOR/'raw'/m['turn_file']) for m in manifests};result={}
 for cohort,(mode,flagged,riseonly) in configs.items():
  rows=[];zs=[];excluded=[];views=protocol['views_rise'] if riseonly else (['TurnBase','TurnContext','ChairMotion'] if mode=='fixed' else protocol['views_both'])
  for oi,m in enumerate(manifests):
   a=data[m['id']];events=audits[m['id']];keys=['rise'] if riseonly else ['rise','sit']
   try:
    if any('window' not in events[k] for k in keys):raise ValueError('missing_required_transition')
    windows={k:(fixed(a,events[k]) if mode=='fixed' else events[k]['window']) for k in keys}
    peaks={k:peak_vertical(a,windows[k]) for k in keys}
    if not flagged and any(v>3 for v in peaks.values()):raise ValueError('chair_vertical_quality')
    rise_end=max(events['rise']['window'][1],windows['rise'][1]);sit_start=min(events['sit']['window'][0],windows.get('sit',events['sit']['window'])[0]) if 'window' in events['sit'] else float(a['time'][-1])
    if rise_end>=m['turn_start'] or sit_start<=m['turn_end']:raise ValueError('overlapping_transitions')
    extra=sum(w[1]-w[0] for w in windows.values());cw=central(a,m,m['turn_observed_seconds']+extra,[rise_end,sit_start]);pre=[cw[0],m['turn_start']];post=[m['turn_end'],cw[1]]
    if min(pre[1]-pre[0],post[1]-post[0])<.2:raise ValueError('context_side_too_short')
    if max(peak_vertical(a,pre),peak_vertical(a,post))>3:raise ValueError('context_vertical_quality')
    base=olddata['Walk3Turn'][oi];cf=np.r_[features(a,pre)[1:],features(a,post)[1:],geom.features(a,cw[0],cw[1],m['axis'])[1:]]
    z={'TurnBase':base,'TurnContext':np.r_[base,cf]};ff=[features(a,windows[k]) for k in keys]
    if riseonly:z['RiseMotion']=np.r_[base,ff[0][1:]]
    else:
     z['ChairTime']=np.r_[base,[f[0] for f in ff]];z['ChairMotion']=np.r_[base,*[f[1:] for f in ff]];z['ChairAll']=np.r_[base,*ff]
    assert all(np.isfinite(v).all() for v in z.values())
    rows.append({**m,'prior_sample':oi,'phase_windows':windows,'phase_max_vertical_speed':peaks,'flagged_vertical':any(v>3 for v in peaks.values()),'context_window':cw,'context_pre':pre,'context_post':post,'observed_seconds':3+m['turn_observed_seconds']+extra});zs.append(z)
   except ValueError as e:excluded.append({'id':m['id'],'target':m['target'],'reason':str(e)})
  if not rows:raise RuntimeError('empty cohort')
  mapping={m['prior_sample']:i for i,m in enumerate(rows)};plan=[]
  for rep in json.loads((PRIOR/'geometry/splits.json').read_text()):
   ss=[]
   for f in rep['splits']:
    remap=lambda arr:[mapping[i] for i in arr if i in mapping]
    ss.append({'fold':f['fold'],'train':remap(f['train']),'test':remap(f['test']),'inner':[{'train':remap(v['train']),'validation':remap(v['validation'])} for v in f['inner']]})
   plan.append({**rep,'splits':ss})
  save(cohort+'/manifest.json',rows);save(cohort+'/splits.json',plan);save(cohort+'/excluded.json',excluded)
  np.savez_compressed(STUDY/cohort/'input.npz',ids=np.array([r['id'] for r in rows]),y=np.array([r['target'] for r in rows]),**{v:np.stack([z[v] for z in zs]) for v in views})
  result[cohort]={'people':len(rows),'long_people':sum(r['target']>=20 for r in rows),'flagged':sum(r['flagged_vertical'] for r in rows),'views':views,'long_ids':[r['id'] for r in rows if r['target']>=20],'excluded_long':[r for r in excluded if r['target']>=20],'observation_range':[min(r['observed_seconds'] for r in rows),max(r['observed_seconds'] for r in rows)],'hashes':{n:sha(STUDY/cohort/n) for n in ['manifest.json','splits.json','input.npz','excluded.json']}}
 save('preparation.json',{'cohorts':result,'protocol_sha256':sha(STUDY/'protocol.json'),'build_sha256':sha(STUDY/'build.py')});print(json.dumps(result),flush=True)
if __name__=='__main__':main()
