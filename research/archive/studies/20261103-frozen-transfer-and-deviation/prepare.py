from pathlib import Path
import sys,json,hashlib,importlib.util,warnings,zipfile,re,time
HERE=Path(__file__).resolve().parent
ROOT=Path('D:/Projects/D_P');BIG=Path('D:/gait-research-data/D_P/frozen-transfer-and-deviation-20261103')
BASE=ROOT/'studies/20261103-reference188-pilot'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proj=module('proj',ROOT/'studies/20261103-reference188-projection-pilot/project.py')
np=proj.np
sys.path.insert(0,str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'))
import preprocess as pre
ns=pre.source_functions()
JOINTS={12:'LFEP',9:'RFEP',13:'LFEO',10:'RFEO',14:'LTIO',11:'RTIO',19:'LTOE',22:'RTOE'}
KEEP=[0,1,2,3,4,5,8,11]
def normid(s):
 m=re.fullmatch(r'(SUBJ|TVC)0*(\d+)',s.upper());return m[1]+str(int(m[2])) if m else s

def main():
 protocol=read(HERE/'protocol.json');assert protocol['frozen_before_scores']
 linkage=read(BASE/'person-linkage.json');persons={normid(r['supplement_candidate_ids'][0]):dict(cohort=g,mat_index=r['mat_index'],age=r['values']['Age']) for g in ['healthy','stroke'] for r in linkage[g]['rows']}
 rows=[];windows=[];owners=[];starts=[];vectors=[];vector_ids=[];pilot=set(['SUBJ1','SUBJ2','SUBJ3','TVC4','TVC8','TVC9']);source_audit=[]
 for source in read(HERE/'downloads.json'):
  assert source['verified'] and sha(source['path'])==source['sha256']
  with zipfile.ZipFile(source['path']) as z:
   for entry in z.infolist():
    parts=entry.filename.split('/');person=next((normid(p) for p in parts[:-1] if normid(p) in persons),None)
    if not person or not entry.filename.lower().endswith('.c3d'):continue
    info=dict(person=person,member=entry.filename,archive=source['path'])
    try:
     data=z.read(entry);tmp=BIG/'current.c3d';tmp.write_bytes(data)
     c=proj.ezc3d.c3d(str(tmp));labels=[n.split(':')[-1].strip() for n in c['parameters']['POINT']['LABELS']['value']]
     assert len(set(labels))==len(labels),'duplicate_labels'
     required=list(JOINTS.values())+['LASI','RASI','LHEE','RHEE']
     missing=[n for n in required if n not in labels];assert not missing,'missing:'+','.join(missing)
     assert c['parameters']['POINT']['UNITS']['value']==['mm'],'wrong_units'
     xyz=np.full((c['data']['points'].shape[2],len(proj.NAMES),3),np.nan)
     for n in required:
      j=labels.index(n);v=c['data']['points'][:3,j,:].T/1000;v[c['data']['meta_points']['residuals'][0,j,:]<0]=np.nan;xyz[:,proj.NAMES.index(n)]=v
     rate=float(c['header']['points']['frame_rate']);t,xyz=proj.resample(xyz,rate)
     assert len(t)>=90,'shorter_than90'
     local,_,_=proj.align(xyz)
     info.update(frames=len(t),source_sha256=hashlib.sha256(data).hexdigest(),source_rate=rate)
     cases=[('side_pan',0)]
     if person in pilot:cases=[(cam,offset) for cam in proj.CONDITIONS for offset in [0,30,60]]
     for camera,offset in cases:
      modified=local.copy()
      for side in 'LR':
       ti=proj.NAMES.index(side+'TOE');hi=proj.NAMES.index(side+'HEE');direction=local[:,ti]-local[:,hi];direction/=np.linalg.norm(direction,axis=1)[:,None];modified[:,ti]+=offset/1000*direction
      uv,rot,pos,depth=proj.project(modified,camera);raw=np.full((len(t),25,3),np.nan)
      for j,n in JOINTS.items():raw[:,j,:2]=uv[:,proj.NAMES.index(n)]
      coords=raw[:,list(JOINTS),:2];inside=np.isfinite(coords).all(2)&(coords[:,:,0]>=0)&(coords[:,:,0]<640)&(coords[:,:,1]>=0)&(coords[:,:,1]<480)
      with warnings.catch_warnings():
       warnings.simplefilter('ignore',RuntimeWarning);processed=ns['process_video_and_add_cols'](raw.reshape(len(t),75),swap_orientation=False)
      feat=np.stack([processed[:,pre.LEFT],processed[:,pre.RIGHT]])
      accepted=[s for s in range(0,500-124,31) if s+124<=len(t) and inside[s:s+124].all() and np.isfinite(feat[:,s:s+124]).all()]
      row={**info,'camera':camera,'toe_offset_mm':offset,'starts':accepted,'primary':camera=='side_pan' and offset==0}
      ix=len(rows);rows.append(row)
      for start in accepted:windows.append(feat[:,start:start+124].astype('float32'));owners.append(ix);starts.append(start)
      if row['primary']:
       # Common-joint summaries never use assumed toe positions. Every selected frame is observed and in frame.
       common=feat[:,:,KEEP];valid=inside[:,:6].all(1)&np.isfinite(common).all(axis=(0,2));runs=[]
       for s in range(0,len(t)-90+1,31):
        if valid[s:s+90].all():runs.extend(range(s,s+90))
       if runs:
        indices=np.unique(runs);v=common[:,indices];adj=indices[1:]-indices[:-1]==1;deriv=np.diff(v,axis=1)[:,adj]*30
        stats=np.stack([v.mean(1),v.std(1),*np.quantile(v,[.05,.25,.5,.75,.95],axis=1),np.sqrt(np.mean(deriv**2,axis=1))],axis=-1)
        assert stats.shape==(2,8,8) and np.isfinite(stats).all()
        vectors.append(stats.ravel());vector_ids.append(person)
     info['status']='processed'
    except (AssertionError,ValueError,IndexError) as exc:info.update(status='excluded',reason=str(exc))
    source_audit.append(info)
    if len(source_audit)%50==0:print('sources',len(source_audit),'windows',len(windows),'people',len(set(vector_ids)),flush=True)
 unique=sorted(set(vector_ids));vx=np.array(vectors);vi=np.array(vector_ids);px=np.stack([vx[vi==p].mean(0) for p in unique])
 np.savez_compressed(BIG/'features.npz',x=np.stack(windows),owner=np.array(owners),start=np.array(starts),person_id=np.array(unique),person_features=px)
 save(HERE/'rows.json',rows);save(HERE/'source-audit.json',source_audit);save(HERE/'people.json',[dict(id=p,**persons[p],feature_trials=int((vi==p).sum())) for p in unique])
 result=dict(protocol_sha256=sha(HERE/'protocol.json'),features_sha256=sha(BIG/'features.npz'),source_files=len(source_audit),processed=sum(r['status']=='processed' for r in source_audit),excluded=sum(r['status']=='excluded' for r in source_audit),people=len(unique),windows=len(windows),primary_records=sum(r['primary'] and bool(r['starts']) for r in rows),primary_people=len(set(r['person'] for r in rows if r['primary'] and r['starts'])),optimizer_steps=0,occlusion_modeled=False,anatomical_toe_equivalence=False,original_cp_timebase_verified=False)
 save(HERE/'preparation.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
