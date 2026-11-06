"""Outcome-blind short-window GSTRIDE preparation and locked cohort extension."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,sys
import numpy as np,joblib
from sklearn.model_selection import StratifiedKFold
import sensor_loop as s
HERE=s.HERE;HEAVY=s.HEAVY;G=HEAVY/'gstride'
def freeze():
    assert not (HERE/'extension-protocol.json').exists()
    s.save(HERE/'extension-protocol.json',{'code_sha256':s.sha(__file__),'sensor_code_sha256':s.sha(HERE/'sensor_loop.py'),
      'workbook_sha256':s.sha(G/'GSTRIDE_DDBB.xlsx'),'workbook_cells_sha256':s.sha(G/'workbook-cells.json'),
      'parameter_receipt_sha256':s.sha(HERE/'gstride-parameters-acquisition.json'),'event_receipt_sha256':s.sha(HERE/'gstride-segmentation-acquisition.json'),
      'source':'https://zenodo.org/records/17052815','advertised_people':163,'processed_files_expected':158,
      'question':'Does a camera-compatible feature predictor generalize to new older people, and does adding this cohort help held-out people?',
      'features':['speed','stride_time','swing_fraction'],'units':['m/s','seconds','fraction0to1'],
      'selection':'First valid straight stride in chronological index order with index-matched HS events; retain consecutive strides sharing exact HS boundary while total CycleDuration<=4s. No skipped gaps. At leastone complete stride.',
      'validity':'0.6<=cycle<=3s,0.05<=speed<=2.5m/s,0.1<=length<=3m,0<=swing<=100percent,abs(turningangle)<20degrees. Outcome-blind. No threshold relaxation.',
      'summary':'Arithmetic mean of permitted stride quantities over selected<=4s run; Per_Swing/100. No whole-record means, cadence, diagnosis, SPPB,4mclinicalspeed,TUGtaskduration or missingness predictors',
      'label':'Workbook AD: Time duration TUG test(s), finite>0, linked by uniqueB participantVNNN ID. Missing/invalid values excluded with reason, not setzero',
      'modes':{'external_zero_adaptation':'Mean prediction of five frozen FRAILPOL outer-fold stacked models, no GSTRIDE labels used in fitting/choice',
        'local_GSTRIDE':'GSTRIDE-only5person folds(seed20261104) with3innerfolds, same candidates and fixed stack',
        'pooled':'Preserve FRAILPOL originalfold and GSTRIDE fixedfold; eachfold both heldout groups excluded. Fit on other people from both cohorts and report cohorts separately'},
      'learning':'Same12 frozen candidate configurations as sensor loop; fixed nonnegative Ridgealpha10 stack. No external-score tuning or choosing the best mode.',
      'semantics_limits':'FRAILPOL is bilateral short walking withinTUG; GSTRIDE is one-foot independent longer walking. Retained nominal quantities have different event detectors and anchors; units and algebra checked, not paired measurement equivalence.',
      'scope':'New sensor-feature cohort, not newRGB/MediaPipevalidation. Once pooled/localCV is run, GSTRIDE also becomes development data. External first-pass predictions kept separately.',
      'evaluation':'MAE,p95,TUG>=20 errors,source-only versus pooled pairedpersonCI. Original594 heldout people never enter their source or pooled fold training.'})
def prepare():
    prot=json.loads((HERE/'extension-protocol.json').read_text());assert s.sha(__file__)==prot['code_sha256']
    assert s.sha(G/'GSTRIDE_DDBB.xlsx')==prot['workbook_sha256'] and s.sha(G/'workbook-cells.json')==prot['workbook_cells_sha256']
    cells=next(iter(json.loads((G/'workbook-cells.json').read_text()).values()))
    def val(row,col):return next((v for k,v in row.items() if ''.join(c for c in k if c.isalpha())==col),None)
    clinical={}
    for row in cells[3:]:
        pid=val(row,'B');assert isinstance(pid,str) and pid not in clinical
        clinical[pid]={'TUG':val(row,'AD'),'SPPB':val(row,'AC'),'age_band':val(row,'J')}
    rows=[];excluded=[];nfiles=0
    for pid,lab in sorted(clinical.items()):
        path=G/'processed/_GAIT/_PARAMETERS'/f'{pid}_GAIT_PARAMETERS.csv'
        ep=G/'processed/_GAIT/_SEGMENTATION'/f'{pid}_GAIT_SEGMENTATION.csv'
        if not path.exists() or not ep.exists():excluded.append({'id':pid,'reason':'no processed gait and event files'});continue
        nfiles+=1
        params=list(csv.DictReader(path.open(encoding='utf-8-sig')));events=list(csv.DictReader(ep.open(encoding='utf-8-sig')))
        pm={int(float(r['index'])):r for r in params};em={int(float(r['index'])):r for r in events}
        assert len(pm)==len(params) and len(em)==len(events)
        matched=[]
        for idx in sorted(pm):
            if idx not in em:continue
            r=pm[idx];ev=em[idx]
            try:
                cd=float(r['CycleDuration[s]']);sp=float(r['Stride_Velocity[m/s]']);sw=float(r['Per_Swing[%CD]'])/100
                length=float(r['StrideLength[m]']);turn=float(r['TurningAngle[degrees]']);start=int(float(ev['HS_ini[sample]']));end=int(float(ev['HS_fin[sample]']));toe=int(float(ev['TO[sample]']))
            except (ValueError,TypeError):continue
            valid=np.isfinite([cd,sp,sw,length,turn]).all() and .6<=cd<=3 and .05<=sp<=2.5 and .1<=length<=3 and 0<=sw<=1 and abs(turn)<20 and start<toe<end
            if valid:
                assert abs(sp-length/cd)<1e-7
                assert abs(sw-(end-toe)/(end-start))<1e-6
            matched.append({'index':idx,'valid':bool(valid),'cycle':cd,'speed':sp,'swing':sw,'start':start,'end':end})
        chosen=None
        for k,r in enumerate(matched):
            if not r['valid']:continue
            selected=[r];duration=r['cycle']
            for nx in matched[k+1:]:
                if not nx['valid'] or nx['index']!=selected[-1]['index']+1 or nx['start']!=selected[-1]['end'] or duration+nx['cycle']>4+1e-9:break
                selected.append(nx);duration+=nx['cycle']
            chosen=selected;break
        if not chosen:excluded.append({'id':pid,'reason':'no valid complete straight stride within4s'});continue
        target=lab['TUG']
        if not isinstance(target,(int,float)) or not np.isfinite(target) or target<=0:excluded.append({'id':pid,'reason':'missing positive clinical TUG'});continue
        rows.append({'id':pid,'y':float(target),'features':[float(np.mean([r[n] for r in chosen])) for n in ['speed','cycle','swing']],
          'stride_indices':[r['index'] for r in chosen],'start_sample':chosen[0]['start'],'end_sample':chosen[-1]['end'],
          'duration_seconds':sum(r['cycle'] for r in chosen),'clinical':lab,'parameters_sha256':s.sha(path),'events_sha256':s.sha(ep)})
    assert nfiles==158 and len(clinical)==163
    y=np.array([r['y'] for r in rows]);X=np.array([r['features'] for r in rows]);ids=np.array([r['id'] for r in rows])
    fold=np.zeros(len(y),int)
    for f,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261104).split(ids,np.digitize(y,[12,20]))):fold[te]=f
    np.savez_compressed(G/'short-window-input.npz',X=X,y=y,ids=ids,fold=fold)
    s.save(HERE/'gstride-short-window-audit.json',{'included':rows,'excluded':excluded,'people':len(y),'files_with_events':nfiles,
      'TUG20plus':int((y>=20).sum()),'TUG_range':[float(y.min()),float(y.max())],
      'duration_range':[min(r['duration_seconds'] for r in rows),max(r['duration_seconds'] for r in rows)],'input_sha256':s.sha(G/'short-window-input.npz')})
    print('GSTRIDE_PREPARED',len(y),'PEOPLE',int((y>=20).sum()),'HIGH_TUG',flush=True)
def external():
    audit=json.loads((HERE/'gstride-short-window-audit.json').read_text());assert s.sha(G/'short-window-input.npz')==audit['input_sha256']
    d=np.load(G/'short-window-input.npz');rec=json.loads((HERE/'sensor-receipt.json').read_text());pred={};models=[]
    for view in ['speed','core3']:
        X=d['X'][:,:1] if view=='speed' else d['X'];foldpred=[]
        for r in rec:
            if r['view']!=view:continue
            mat=[]
            for cfg in range(len(s.CONFIGS)):
                item=next(m for m in r['models'] if m['candidate']==cfg and m['inner'] is None)
                path=HEAVY/'sensor-models'/item['file'];assert s.sha(path)==item['sha256'];m=joblib.load(path);mat.append(s.predict(m,X))
            meta=joblib.load(HEAVY/'sensor-models'/r['meta'])['stacked'];p=s.pred_stack(meta,np.column_stack(mat));foldpred.append(p)
            models.append({'view':view,'fold':r['fold'],'source_context_sha256':r['context_sha256']})
        assert len(foldpred)==5;pred[view]=np.mean(foldpred,axis=0)
    np.savez_compressed(G/'external-zero-adaptation.npz',y=d['y'],ids=d['ids'],**pred)
    s.save(HERE/'gstride-external-receipt.json',{'models':models,'output_sha256':s.sha(G/'external-zero-adaptation.npz'),'fitting_on_GSTRIDE':False})
    print('GSTRIDE_FIRST_EXTERNAL_PASS_SAVED',flush=True)
def train():
    prot=json.loads((HERE/'extension-protocol.json').read_text());assert s.sha(__file__)==prot['code_sha256']
    assert (HERE/'gstride-external-receipt.json').exists(),'Preserve external first pass before local/pool fitting'
    gd=np.load(G/'short-window-input.npz');fd,FX,sets=s.load();records=[];(HEAVY/'extension-models').mkdir(exist_ok=True)
    for mode in ['local_GSTRIDE','pooled']:
      for view in ['speed','core3']:
        GX=gd['X'][:,:1] if view=='speed' else gd['X']
        if mode=='local_GSTRIDE':X=GX;y=gd['y'];ids=gd['ids'];folds=gd['fold'];cohort=np.array(['G']*len(y))
        else:
            X=np.concatenate([FX[view],GX]);y=np.r_[fd['y'],gd['y']];ids=np.r_[np.array(['F'+str(x) for x in fd['ids']]),gd['ids']]
            folds=np.r_[fd['fold'],gd['fold']];cohort=np.r_[np.array(['F']*len(fd['y'])),np.array(['G']*len(gd['y']))]
        for f in range(5):
          tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f);A=np.full((len(tr),len(s.CONFIGS)),np.nan);T=np.empty((len(te),len(s.CONFIGS)));models=[]
          splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])))
          for k,cfg in enumerate(s.CONFIGS):
            for ii,(it,iv) in enumerate(splits):
                m=s.fit(X[tr[it]],y[tr[it]],cfg,sets[view]);path=HEAVY/'extension-models'/f'{mode}-{view}-f{f}-c{k}-i{ii}.joblib';joblib.dump(m,path,compress=3)
                A[iv,k]=s.predict(m,X[tr[iv]]);models.append({'file':path.name,'sha256':s.sha(path),'candidate':k,'inner':ii,'train':tr[it].tolist(),'test':tr[iv].tolist()})
            m=s.fit(X[tr],y[tr],cfg,sets[view]);path=HEAVY/'extension-models'/f'{mode}-{view}-f{f}-c{k}-final.joblib';joblib.dump(m,path,compress=3)
            T[:,k]=s.predict(m,X[te]);models.append({'file':path.name,'sha256':s.sha(path),'candidate':k,'inner':None,'train':tr.tolist(),'test':te.tolist()})
          meta=s.stack(A,y[tr]);pp=s.pred_stack(meta,T)
          context=HEAVY/'extension-models'/f'{mode}-{view}-f{f}-context.npz';np.savez_compressed(context,A=A,T=T,prediction=pp,train=tr,test=te,ids=ids,y=y,cohort=cohort)
          mp=HEAVY/'extension-models'/f'{mode}-{view}-f{f}-meta.joblib';joblib.dump(meta,mp)
          records.append({'mode':mode,'view':view,'fold':f,'train':tr.tolist(),'test':te.tolist(),'models':models,'context':context.name,'context_sha256':s.sha(context),'meta':mp.name,'meta_sha256':s.sha(mp)})
          s.save(HERE/'extension-receipt.json',records);print('EXTENSION_CASE',mode,view,f,flush=True)
    print('EXTENSION_TRAIN_COMPLETE',len(records),'CASES',flush=True)
if __name__=='__main__':{'freeze':freeze,'prepare':prepare,'external':external,'train':train}[sys.argv[1]]()
