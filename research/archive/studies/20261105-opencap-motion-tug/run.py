"""Frozen person-disjoint learned-motion TUG comparison, local CPU."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,sys,time
import numpy as np,joblib
from scipy.optimize import nnls
import prepare as p
import learning as l
H=p.H;B=p.B

def freeze():
    assert not(H/'protocol.json').exists();audit=p.read(H/'input-audit.json');assert audit['status']=='passed'
    bindings={}
    selections=p.read(p.OLD/'selection.json')
    for row in selections:
        if row['repeat']!=0 or row['view'] not in ['speed_only','speed_dynamics','speed_plus_landmarks']:continue
        stem=f"r0-f{row['fold']}-{row['view']}"
        for ext in ['.joblib','-inner.npz']:
            path=p.DATA/'models'/(stem+ext);bindings[str(path)]=p.sha(path)
    p.save(H/'protocol.json',{
      'question':'Do learned side-view short-walk motion and joint+feature learning improve same-visit actual clinical TUG seconds?',
      'authorization':'User approved return to TUG and same-person speed/features/learned-joint comparison, engineering improvement targets.',
      'code_sha256':{name:p.sha(H/name) for name in ['prepare.py','learning.py','run.py']},'input_sha256':audit['input_sha256'],'original_splits_sha256':p.sha(p.OLD/'effective-splits.json'),'original_results_sha256':p.sha(p.OLD/'results.json'),'source_baseline_bindings':bindings,
      'people':297,'age65plus':34,'TUG20plus':9,'source':'https://zenodo.org/records/22309771',
      'target':'samepersonsamevisit tft_tug seconds; separate3secondnormal10mwalkingsegment,notTUGduration asinput.',
      'input':'Exactold91samplesat30Hz,3sec,SG7/order2,idealcalibratedsideprojection.16landmarkswithouttoes;neck/midhipderivablefromMediaPipe,heelsincluded.32hip-centredcoordinates+2pelvisdynamicchannels,body-scaled=34channels. Sameoldmetric speed.',
      'forbidden':'Age,sex,diagnosis,otherclinicalscores,TUGmotion,testduration,augmentedanatomicalmarkers,transversedepth,extraqualitymetadatasequencechannels.',
      'arms':{'A':'Previouslyfrozenselectedspeed-onlytabularbaseline,replay','A2':'Previousspeeddynamicstabularbaseline,replay','B':'Previous64featuretabularbaseline,replay','C':'TCN34x91sequence+speed','D':'SameTCN34x91sequence+all64features','R':'256fixedrandomtemporalfilters(max/positivefraction)+64features,Ridgealpha10/100/1000','E':'FixednonnegativeRidge1blendofinnerOOFB/C/D/R,train-onlycenteringandscaling'},
      'TCN':{'grid':l.TCN,'architecture':'ResidualdilatedConv1Dwidth16/32,dilations1/2/4,mean+maxpool,32unithead;noageinput','optimizer':'AdamWlr.001,weightdecay.05,Huberbeta.5standardizedtarget,batch64,gradientclip5,trainingsequenceGaussiannoise.005','epochs':'max100,patience15;innerbestMAEepoch,finalmedian3innerbestepochs;noouterstopping','target':'rawTUGorresidualabove6/max(speed,.1).6/v isillustrativeanchor,notmeasuredwalkingcomponent;targetmean/stdtrainonly','normalization':'Sequencechannelandfeaturemedians/scalersfitonlytrainingpeople;missingfeatureimputationisnotphysicalzeromeasurement'},
      'splits':'Originalrepeat0fivepersonouterfoldsandoriginal3innerfolds,reuseddevelopmentcohort.Eachpersonexactlyoneouterprediction.Allarmsuseidenticalpeople/windows. Other4repeatsnotrununlessprimarypromising.',
      'counts':{'internal_C_D':120,'internal_R':45,'final_C_D_R':15,'blends':5,'old_baseline_final_replays':15},
      'diagnostic':'Test-timepermutationof91frames,C/Donly,descriptive,notatrainedcomparison. RecordallpeopleAE>1sec,scorebands,bias,highTUGcaseprofiles.',
      'goals':'TUGMAE<=1sec,within2sec>=80percent,relativeMAEreduction>=20percentvsA;allandage65plus;paired95CIvsAandA2upper<0. EngineeringnotclinicalMDC.',
      'continuation':'Ifanonbaselineprimarymethodpassesallgoalsorpaired95CIvsBupper<0,runallfourremainingoldpartitionswithsameprotocolasconditionaldevelopmentstabilitycheck. Otherwisecloseprimarywithfailurediagnosis;donotkeepouter-tuninguntilanumberpasses.',
      'verification':'Every165internaland15finalmodelSHA/reloadprediction,training-onlynorm/targetstats,personboundary,innerselection,epochmedian;CPU64TCNtolerance2e-5;baseline15modelsreplay;blendNNLSrebuild;independentmetricsandpaired10000personbootstrap. Neverrelaxfailuretolerance.',
      'storage':'LocalCPU1thread;noexternalGPU/upload/newdownloads.Expectedless1GBinD:/gait-research-data/D_P/opencap-motion-tug-20261105;~50GBfreeatstart.',
      'limits':'Syntheticidealviewofmulticameravideo-derived3D,notactualphone/MediaPipemeasurementaccuracy;neuromusculardatasetnotgeneralelderly;34olderand9slow;repeated-usecohortnotindependentclinicalvalidation. ExistingB/D/Rincludeavailability-derivedfeaturesfromold64schema,nophoneaccuracyclaim.'})
    print('PROTOCOL_FROZEN',flush=True)

def check():
    prot=p.read(H/'protocol.json')
    for name,digest in prot['code_sha256'].items():assert p.sha(H/name)==digest
    assert p.sha(B/'input.npz')==prot['input_sha256'] and p.sha(p.OLD/'effective-splits.json')==prot['original_splits_sha256']
    for path,digest in prot['source_baseline_bindings'].items():assert p.sha(path)==digest
    return prot,np.load(B/'input.npz'),p.read(p.OLD/'effective-splits.json')[0]['splits']

def old_models(d,fold):
    result={};tr=np.array(fold['train']);te=np.array(fold['test']);choices=p.read(p.OLD/'selection.json')
    for arm,view,cols in [('A','speed_only',[0]),('A2','speed_dynamics',list(range(4))),('B','speed_plus_landmarks',list(range(64)))]:
        row=next(r for r in choices if r['repeat']==0 and r['fold']==fold['fold'] and r['view']==view);np.testing.assert_array_equal(row['train'],tr);np.testing.assert_array_equal(row['test'],te)
        stem=f"r0-f{fold['fold']}-{view}";path=p.DATA/'models'/(stem+'.joblib');m=joblib.load(path);assert p.sha(path)==row['model_sha256'];inner=p.DATA/'models'/(stem+'-inner.npz');ip=np.load(inner);errors={k:float(abs(ip[k]-d['y'][tr]).mean()) for k in ip.files};assert min(errors,key=errors.get)==row['selected']
        for key in errors:assert abs(errors[key]-row['inner_MAE'][key])<1e-12
        result[arm]={'inner':ip[row['selected']],'test':m.predict(d['X'][te][:,cols]),'model':str(path),'inner_path':str(inner),'selected':row['selected']}
    return result

def predict(m,d,idx,arm,double=False):
    X=d['X'][idx,:1] if arm=='C' else d['X'][idx]
    return l.predict_random(m,d['sequence'][idx],X) if arm=='R' else l.predict_tcn(m,d['sequence'][idx],X,double)

def pilot():
    _,d,folds=check();f=folds[0];a=np.array(f['inner'][0]['train']);v=np.array(f['inner'][0]['validation']);outer=np.array(f['test']);assert not set(a)&set(outer) and not set(v)&set(outer)
    start=time.monotonic();m=l.fit_tcn(d['sequence'][a],d['X'][a,:1],d['y'][a],l.TCN[0],20261105,validation=(d['sequence'][v],d['X'][v,:1],d['y'][v]));path=B/'pilot.joblib';joblib.dump(m,path,compress=3);reloaded=joblib.load(path)
    pred=l.predict_tcn(reloaded,d['sequence'][v],d['X'][v,:1]);double=l.predict_tcn(reloaded,d['sequence'][v],d['X'][v,:1],True);diff=float(max(abs(pred-double)));assert diff<=2e-5
    elapsed=time.monotonic()-start;p.save(H/'pilot-verification.json',{'status':'passed','train_people':len(a),'validation_people':len(v),'outer_people_used':0,'epochs':m['epochs'],'epochs_run':len(m['history']),'seconds':elapsed,'max_CPU64_difference':diff,'pilot_model_sha256':p.sha(path),'rough_total_CPU_seconds':elapsed*120,'single_fit_estimate_only':True});print('PILOT',round(elapsed,2),m['epochs'],diff,flush=True)

def train():
    _,d,folds=check();assert p.read(H/'pilot-verification.json')['status']=='passed';folder=B/'models';folder.mkdir(exist_ok=True);records=[];start=time.monotonic()
    for fold in folds:
      f=fold['fold'];tr=np.array(fold['train']);te=np.array(fold['test']);innerOOF={};outerpred={};fits=[];final=[];selected={}
      for arm in ['C','D','R']:
        configs=l.RANDOM if arm=='R' else l.TCN;A=np.full((len(tr),len(configs)),np.nan);positions={int(value):i for i,value in enumerate(tr)};epochs={k:[] for k in range(len(configs))}
        for k,cfg in enumerate(configs):
          for j,inner in enumerate(fold['inner']):
            ti=np.array(inner['train']);vi=np.array(inner['validation']);X=d['X'][:,:1] if arm=='C' else d['X']
            if arm=='R':m=l.fit_random(d['sequence'][ti],X[ti],d['y'][ti],cfg)
            else:m=l.fit_tcn(d['sequence'][ti],X[ti],d['y'][ti],cfg,20261105+f*100+k,validation=(d['sequence'][vi],X[vi],d['y'][vi]));epochs[k].append(m['epochs'])
            path=folder/f'f{f}-{arm}-k{k}-i{j}.joblib';joblib.dump(m,path,compress=3);A[[positions[int(i)] for i in vi],k]=predict(m,d,vi,arm)
            fits.append({'arm':arm,'candidate':k,'inner':j,'train':ti.tolist(),'test':vi.tolist(),'path':str(path),'sha256':p.sha(path)})
        assert np.isfinite(A).all();errors=abs(A-d['y'][tr,None]).mean(0);k=int(np.argmin(errors));cfg=configs[k];selected[arm]={'candidate':k,'inner_mae':errors.tolist()};innerOOF[arm]=A[:,k]
        X=d['X'][:,:1] if arm=='C' else d['X']
        if arm=='R':m=l.fit_random(d['sequence'][tr],X[tr],d['y'][tr],cfg)
        else:m=l.fit_tcn(d['sequence'][tr],X[tr],d['y'][tr],cfg,20261105+f*100+k,epochs=max(1,int(np.median(epochs[k]))))
        path=folder/f'f{f}-{arm}-final.joblib';joblib.dump(m,path,compress=3);outerpred[arm]=predict(m,d,te,arm);final.append({'arm':arm,'path':str(path),'sha256':p.sha(path)})
        print('FITTED',f,arm,'seconds',round(time.monotonic()-start,1),flush=True)
      prior=old_models(d,fold)
      for arm in ['A','A2','B']:innerOOF[arm]=prior[arm]['inner'];outerpred[arm]=prior[arm]['test']
      blendnames=['B','C','D','R'];A=np.column_stack([innerOOF[arm] for arm in blendnames]);center=A.mean(0);scale=np.maximum(A.std(0),1e-6);ym=float(d['y'][tr].mean());coef=nnls(np.vstack([(A-center)/scale,np.eye(4)]),np.r_[d['y'][tr]-ym,np.zeros(4)])[0]
      outerpred['E']=np.maximum(0,(np.column_stack([outerpred[arm] for arm in blendnames])-center)/scale@coef+ym)
      blend={'names':blendnames,'center':center,'scale':scale,'coef':coef,'ymean':ym};path=folder/f'f{f}-E.joblib';joblib.dump(blend,path,compress=3)
      context=folder/f'f{f}-context.npz';np.savez_compressed(context,train=tr,test=te,**{f'inner_{arm}':pr for arm,pr in innerOOF.items()},**outerpred)
      records.append({'fold':f,'train':tr.tolist(),'test':te.tolist(),'inner':fits,'final':final,'selected':selected,'blend':str(path),'blend_sha256':p.sha(path),'context':str(context),'context_sha256':p.sha(context)})
      p.save(H/'fit-receipt.json',{'records':records,'protocol_sha256':p.sha(H/'protocol.json'),'input_sha256':p.sha(B/'input.npz'),'elapsed_seconds':time.monotonic()-start})
      print('SAVED_FOLD',f,flush=True)

if __name__=='__main__':{'freeze':freeze,'pilot':pilot,'train':train}[sys.argv[1]]()
