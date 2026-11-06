from pathlib import Path
import sys,json,hashlib,re,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'D:/gait-research-data/D_P/dataset-audits/20261103-reference188-projection-pilot/deps')
import ezc3d,numpy as np
manifest=json.loads((HERE/'manifest.json').read_text());reference=Path('D:/Projects/D_P/outputs/gdi-reference-investigation-20261030/gdi_original_active166_reference.npz')
with np.load(reference) as f:ref={k:f[k] for k in f.files}
def gdi(curve):
 vector=curve.T.reshape(-1);distance=np.linalg.norm(vector@ref['feature_basis']-ref['control_feature_center'])
 return float(100-10*(np.log(distance)-ref['raw_distance_log_mean'])/ref['raw_distance_log_sample_sd'])
def norm(x):return x/np.linalg.norm(x,axis=1)[:,None]
def reportval(x):return dict(min=float(np.nanmin(x)),median=float(np.nanmedian(x)),max=float(np.nanmax(x)))
rows=[];histories=[]
for row in manifest:
 p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 if p.suffix=='.history':
  tree=ET.fromstring(p.read_text());ops=[]
  for op in tree.iter('OperationSubType'):
   pars={a.attrib['name']:a.attrib.get('value') for a in op.iter('Param')};ops.append(dict(name=op.attrib.get('name'),date=pars.get('Date'),output_joint_centers=pars.get('Output Joint Centers'),marker_diameter=pars.get('Marker Diameter')))
  histories.append(dict(person=row['person'],member=row['member'],operations=ops));continue
 if p.suffix!='.c3d':continue
 c=ezc3d.c3d(str(p));pa=c['parameters'];labels=[n.split(':')[-1] for n in pa['POINT']['LABELS']['value']];points=c['data']['points'][:3].transpose(2,1,0);res=c['data']['meta_points']['residuals'][0].T;points[res<0]=np.nan
 def pt(k):return points[:,labels.index(k)]
 static='(0)' in row['member'] or ' Cal ' in row['member'];rate=float(c['header']['points']['frame_rate']);first=int(c['header']['points']['first_frame'])+1
 out=dict(person=row['person'],member=row['member'],static=static,frames100=len(points),frames30=int((len(points)-1)/rate*30)+1,joint_labels_present=all(s+k in labels for s in ['L','R'] for k in ['FEP','FEO','TIO']))
 proc=pa.get('PROCESSING',{});required=['LLegLength','RLegLength','InterAsisDistance','LKneeWidth','RKneeWidth','LAnkleWidth','RAnkleWidth']
 if all(k in proc for k in required):
  val=lambda k:float(proc[k]['value'][0]);C=.115*(val('LLegLength')+val('RLegLength'))/2-15.3;aa=(val('InterAsisDistance') if val('InterAsisDistance')!=0 else float(np.nanmean(np.linalg.norm(pt('LASI')-pt('RASI'),axis=1))))/2
  origin=(pt('LASI')+pt('RASI'))/2;y=norm(pt('LASI')-pt('RASI'));sac=pt('SACR') if 'SACR' in labels else (pt('LPSI')+pt('RPSI'))/2
  x=origin-sac;x=norm(x-(x*y).sum(1)[:,None]*y);z=np.cross(x,y);out['independent_joint_geometry']={}
  for side in ['L','R']:
   at=val(side+'AsisTrocanterDistance') if side+'AsisTrocanterDistance' in proc else 0
   if at==0:at=.1288*val(side+'LegLength')-48.56
   ox=C*np.cos(.5)*np.sin(.314)-(at+7)*np.cos(.314);oy=-(C*np.sin(.5)-aa)*(1 if side=='L' else -1);oz=-C*np.cos(.5)*np.cos(.314)-(at+7)*np.sin(.314)
   hip=origin+ox*x+oy*y+oz*z;hiperror=np.linalg.norm(hip-pt(side+'FEP'),axis=1)
   knee_offset=np.linalg.norm(pt(side+'FEO')-pt(side+'KNE'),axis=1)-(val(side+'KneeWidth')/2+7)
   ankle_offset=np.linalg.norm(pt(side+'TIO')-pt(side+'ANK'),axis=1)-(val(side+'AnkleWidth')/2+7)
   out['independent_joint_geometry'][side]=dict(hip_reconstruction_error_mm=reportval(hiperror),knee_offset_abs_error_mm=reportval(np.abs(knee_offset)),ankle_offset_abs_error_mm=reportval(np.abs(ankle_offset)))
 else:out['independent_joint_geometry']=None
 out['gdi_processing']={k:np.asarray(v['value']).tolist() for k,v in proc.items() if 'GDI' in k.upper()};an=pa.get('ANALYSIS',{});out['gdi_analysis']={an['CONTEXTS']['value'][i]:float(an['VALUES']['value'][i]) for i,n in enumerate(an.get('NAMES',{}).get('value',[])) if n=='GDI'}
 out['gdi_replay']={}
 if not static and 'EVENT' in pa:
  event=pa['EVENT'];times=event['TIMES']['value'][0]*60+event['TIMES']['value'][1]
  for side,context in [('L','Left'),('R','Right')]:
   strikes=sorted(int(round(t*rate))-first for t,n,ctx in zip(times,event['LABELS']['value'],event['CONTEXTS']['value']) if n=='Foot Strike' and ctx==context)
   specs=[('PelvisAngles',j) for j in range(3)]+[('HipAngles',j) for j in range(3)]+[('KneeAngles',0),('AnkleAngles',0),('FootProgressAngles',2)]
   angles=np.column_stack([pt(side+k)[:,j] for k,j in specs]);curves=[];shifts=[];indices=[]
   for start,stop in zip(strikes[:-1],strikes[1:]):
    if start<0 or stop>=len(points) or stop<=start or not np.isfinite(angles[start:stop+1]).all():continue
    ts=np.linspace(start,stop,51);curves.append(np.column_stack([np.interp(ts,np.arange(len(points)),angles[:,j]) for j in range(9)]));indices.append([start,stop])
    if stop+1<len(points):shifts.append(np.column_stack([np.interp(ts+1,np.arange(len(points)),angles[:,j]) for j in range(9)]))
   if curves:
    scores=[gdi(curve) for curve in curves];out['gdi_replay'][context]=dict(cycles=len(curves),cycle_indices=indices,cycle_scores=scores,mean_cycle_gdi=float(np.mean(scores)),gdi_of_mean_curve=gdi(np.mean(curves,axis=0)),plus1_sample_mean_gdi=float(np.mean([gdi(v) for v in shifts])) if shifts else None,analysis_value=out['gdi_analysis'].get(context))
    if context in out['gdi_analysis']:out['gdi_replay'][context]['primary_minus_analysis']=float(np.mean(scores)-out['gdi_analysis'][context])
 rows.append(out)
summary=dict(protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),records=rows,histories=histories,source_gdi_not_used_as_training_labels=True)
(HERE/'joint-gdi-audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
for r in rows:
 if r['static']:continue
 print(r['person'],Path(r['member']).name,'frames',r['frames30'],'hip max',[round(v['hip_reconstruction_error_mm']['max'],5) for v in (r['independent_joint_geometry'] or {}).values()],'GDI replay',[(k,round(v['mean_cycle_gdi'],4),v['analysis_value']) for k,v in r['gdi_replay'].items()])
