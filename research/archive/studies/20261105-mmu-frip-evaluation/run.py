"""Nested person comparisons and predeclared train-only improvement rounds."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,joblib,torch
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import nnls
import prepare as p
import learning as l
H=p.H;B=p.B
def load():return np.load(B/'input.npz')
def freeze():
    assert not(H/'model-protocol.json').exists();d=load()
    p.save(H/'model-protocol.json',dict(code_sha256=p.sha(__file__),library_sha256=p.sha(Path(l.__file__)),input_sha256=p.sha(B/'input.npz'),input_protocol_sha256=p.sha(H/'input-protocol.json'),
      primary={'A_speed':'body-size-normalizedimageplanespeed,same2secwindow','B_features':'speed+56same-window2Dmotiondescriptors','C_learned_movement':'same2D28ch60framejointsequence+sameonespeedscalar,TCN'},
      inner='3 stratifiedpersonfolds seed20261105 onscorebins<6/6..13/>=14 within each fixedoutertrainingfold;fitpreprocessingandtargetnormalizationonlyinnertraining. SelectminimumpooledinnerOOF MAE,firsttie. FinalTCNepoch=medianbestepochsacross3innerfitsforselectedconfig;neverouterstopping.',
      tabular_candidates=l.TAB,temporal_candidates=l.TCN,random_filter_candidates=l.RANDOM,
      improvement_loop={'round1':'A/B Ridge/SVR and CTCN; saveeveryinnerresult','round2':'A/BadditionalLightGBM candidates and Rrandom-temporalfilters+Ridge; noextramodalities','round3':'DnonnegativeRidgealpha1blendofinnerOOF A/B/C/Rpredictions,intercepttrainingmean;fixedregularization,outertestneverused. Includeshandcraftedfeatures,reportascombinedseparatelyfromC.',
        'trigger':'MAE>1pointininnertrainingvalidation triggers cause-review;allprespecifiedarmsretainedregardlessofwhichwins. Outerpredictionssealeduntilallfivefoldsfinished;noadditionaltuningtotheseheldoutresults.'},
      controls='Trainingmedian,trainingmean,age-onlyRidgealpha10 diagnostic (ageispartofclinicalrisk;neveraddedtoprimarymodels); test-time temporalshuffle ofC diagnostic,notnewmodelselection.',
      target='JH-FRATpointsnotseconds. No age,gender,race,fallhistory,medications,mobility,cognition,scorecomponents,fullTUGtimeorinferredTUGdurationprimaryinputs.',
      metrics='MAE,RMSE,bias,R2,Spearman,within1/2/3,largeerrors>3,scorebands<6/6..13/>=14;thresholdpred>=6riskAUROC/balancedaccuracy/recall/specificity secondary;all125and65plus56fromsameheldoutfits;5000pairedperson95CIandMAECI. No clinicalMDCclaim.',
      goals={'MAE_max':1,'within2_min':.8,'relative_MAE_improvement_vs_A_min':.2,'both_all_and_age65plus':True,'paired_delta_CI_upper_below_zero':True},
      replay_tolerance={'fp32_prediction':1e-10,'CPU64_TCN':2e-5,'independent_metrics':1e-10},
      resources='CPUone-thread,noexternalGPUtransfer;allinternal/finalmodelsandcontextsunderlargepath. PilotoneinnerTCN,reload/CPU64check,thenlockedbatch. Singleinitializationperfit;clinicalexternalvalidationabsent.'))
def pilot():
    d=load();tr=np.flatnonzero(d['fold']!=0);it,iv=next(StratifiedKFold(3,shuffle=True,random_state=20261105).split(tr,np.digitize(d['y'][tr],[6,14])));ti=tr[it];vi=tr[iv];started=time.monotonic()
    m=l.fit_tcn(d['sequence'][ti],d['X'][ti,0],d['y'][ti],l.TCN[0],20261105,validation=(d['sequence'][vi],d['X'][vi,0],d['y'][vi]));dest=B/'pilot-tcn.pt';torch.save(m,dest);again=torch.load(dest,weights_only=False);a=l.predict_tcn(again,d['sequence'][vi],d['X'][vi,0]);double=l.predict_tcn(again,d['sequence'][vi],d['X'][vi,0],True);difference=float(max(abs(a-double)));assert difference<=2e-5
    p.save(H/'pilot-verification.json',{'status':'passed','inner_train':len(ti),'inner_validation':len(vi),'outerheldout_used':False,'epochs':m['epochs'],'seconds':time.monotonic()-started,'CPU64_max_difference':difference,'sha256':p.sha(dest),'code_sha256':p.sha(__file__)})
    print('TCN_PILOT_PASSED_SECONDS',round(time.monotonic()-started,2),'EPOCHS',m['epochs'],'CPU64',difference,flush=True)
def train():
    prot=json.loads((H/'model-protocol.json').read_text());assert prot['code_sha256']==p.sha(__file__) and prot['library_sha256']==p.sha(Path(l.__file__)) and prot['input_sha256']==p.sha(B/'input.npz');assert json.loads((H/'pilot-verification.json').read_text())['status']=='passed';assert json.loads((H/'input-verification.json').read_text())['status']=='passed'
    d=load();folder=B/'models';folder.mkdir(exist_ok=True);records=[];started=time.monotonic()
    for fold in range(5):
      tr=np.flatnonzero(d['fold']!=fold);te=np.flatnonzero(d['fold']==fold);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261105).split(tr,np.digitize(d['y'][tr],[6,14])));innerrecords=[];selected={};oof={};testpred={};final=[]
      for arm in ['A','B','C','R']:
        configs=l.TAB if arm in ['A','B'] else l.TCN if arm=='C' else l.RANDOM;A=np.empty((len(tr),len(configs)));epochs={}
        for k,cfg in enumerate(configs):
          epochs[k]=[]
          for j,(it,iv) in enumerate(splits):
            ti=tr[it];vi=tr[iv];seed=20261105+fold*100+k
            if arm in ['A','B']:
                X=d['X'][:,:1] if arm=='A' else d['X'];m=l.fit_tab(X[ti],d['y'][ti],cfg);pp=l.predict_tab(m,X[vi])
            elif arm=='C':
                m=l.fit_tcn(d['sequence'][ti],d['X'][ti,0],d['y'][ti],cfg,seed,validation=(d['sequence'][vi],d['X'][vi,0],d['y'][vi]));pp=l.predict_tcn(m,d['sequence'][vi],d['X'][vi,0]);epochs[k].append(m['epochs'])
            else:m=l.fit_random(d['sequence'][ti],d['X'][ti,0],d['y'][ti],cfg);pp=l.predict_random(m,d['sequence'][vi],d['X'][vi,0])
            path=folder/f'f{fold}-{arm}-k{k}-i{j}.joblib';joblib.dump(m,path,compress=3);A[iv,k]=pp;innerrecords.append({'arm':arm,'candidate':k,'inner':j,'train':ti.tolist(),'test':vi.tolist(),'file':str(path),'sha256':p.sha(path),'epochs':m.get('epochs')})
          print('INNER_CONFIG_DONE',fold,arm,k,'SECONDS',round(time.monotonic()-started,1),flush=True)
        errors=abs(A-d['y'][tr,None]).mean(0);best=int(np.argmin(errors));selected[arm]={'candidate':best,'config':configs[best],'inner_mae':float(errors[best]),'all_candidate_inner_mae':errors.tolist(),'epochs':epochs.get(best)};oof[arm]=A[:,best]
        cfg=configs[best]
        if arm in ['A','B']:
            X=d['X'][:,:1] if arm=='A' else d['X'];m=l.fit_tab(X[tr],d['y'][tr],cfg);pp=l.predict_tab(m,X[te])
        elif arm=='C':
            ep=max(1,int(np.median(epochs[best])));m=l.fit_tcn(d['sequence'][tr],d['X'][tr,0],d['y'][tr],cfg,20261105+fold*100+best,epochs=ep);pp=l.predict_tcn(m,d['sequence'][te],d['X'][te,0])
        else:m=l.fit_random(d['sequence'][tr],d['X'][tr,0],d['y'][tr],cfg);pp=l.predict_random(m,d['sequence'][te],d['X'][te,0])
        path=folder/f'f{fold}-{arm}-final.joblib';joblib.dump(m,path,compress=3);final.append({'arm':arm,'file':str(path),'sha256':p.sha(path)});testpred[arm]=pp
      A=np.column_stack([oof[a] for a in ['A','B','C','R']]);T=np.column_stack([testpred[a] for a in ['A','B','C','R']]);center=A.mean(0);scale=np.maximum(A.std(0),1e-6);ym=float(d['y'][tr].mean());z=(A-center)/scale;coef=nnls(np.vstack([z,np.eye(4)]),np.r_[d['y'][tr]-ym,np.zeros(4)])[0];blend={'center':center,'scale':scale,'ymean':ym,'coef':coef};testpred['D']=np.maximum(0,(T-center)/scale@coef+ym)
      age=l.fit_tab(d['age'][tr,None],d['y'][tr],{'kind':'ridge','alpha':10});path=folder/f'f{fold}-age-final.joblib';joblib.dump(age,path,compress=3);final.append({'arm':'age','file':str(path),'sha256':p.sha(path)});testpred['age']=l.predict_tab(age,d['age'][te,None]);testpred['median']=np.repeat(np.median(d['y'][tr]),len(te));testpred['mean']=np.repeat(d['y'][tr].mean(),len(te))
      cm=joblib.load(next(r['file'] for r in final if r['arm']=='C'));order=np.random.default_rng(20261105).permutation(60);testpred['C_shuffled']=l.predict_tcn(cm,d['sequence'][te][:,:,order],d['X'][te,0])
      context=folder/f'f{fold}-context.npz';np.savez_compressed(context,train=tr,test=te,innerOOF=A,outerbases=T,**testpred)
      meta=folder/f'f{fold}-blend.joblib';joblib.dump(blend,meta)
      records.append({'fold':fold,'train':tr.tolist(),'test':te.tolist(),'selected':selected,'inner':innerrecords,'final':final,'context':str(context),'context_sha256':p.sha(context),'blend':str(meta),'blend_sha256':p.sha(meta)})
      p.save(H/'fit-receipt.json',{'folds':records,'code_sha256':p.sha(__file__),'library_sha256':p.sha(Path(l.__file__)),'input_sha256':p.sha(B/'input.npz'),'seconds':time.monotonic()-started});print('OUTER_FOLD_COMPLETE',fold,'SECONDS',round(time.monotonic()-started,1),flush=True)
    print('ALL_FIVE_FOLDS_COMPLETE',round(time.monotonic()-started,1),flush=True)
if __name__=='__main__':{'freeze':freeze,'pilot':pilot,'train':train}[sys.argv[1]]()
