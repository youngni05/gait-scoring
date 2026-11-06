"""Declared exploratory follow-up: same-window estimated3D, preserving round1."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,csv,sys,time,copy
from pathlib import Path
import numpy as np,joblib,torch
from torch import nn
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold
import prepare as p
import learning as l
H=p.H;B=p.B
def freeze():
    assert not(H/'round2-protocol.json').exists()
    p.save(H/'round2-protocol.json',{'code_sha256':p.sha(__file__),'round1_results_sha256':p.sha(H/'results.json'),'input_sha256':p.sha(B/'input.npz'),
      'status':'Exploratoryimprovementafterround1OOFseen. Reused5folddevelopment,notnewexternalconfirmation. Originalthreeconditions/resultsfrozenandretained.',
      'matching':'Reuse125exactselected60frames. In each2Dframefindtrackedsourcepersonbyexactcoordinateequality;match3DrecordbymaximumYOLO-boxIoU>=.5. No clinicalscore/frameconfusion.',
      'inputs':{'B_2D3D':'Original57featuresplus72jointcoordinateSTD+24jointvelocityRMS and6jointangle4stats=1203Drelativefeatures. Noage/clinicalinputs. Estimated3DHybrIKnotgroundtruth.',
        'C_3D':'24x3jointchannels,hip/rootcentred,median3Dneck-to-midanklescale,30Hzsame60frames,same2Dspeed. SamefourTCNconfigs,3innerfoldselectionandinternalepochmedian.',
        'S_provider_diagnostic':'Provider72fullTUG2D/3Dfeatures,NOclinicalvariables. Fulltestcontext,unverifiedmetriccalibration =>notshortwalkingorphonevalidation. Diagnoseinformationlossinoursamewindowextraction.'},
      'models':'B/S same9Ridge/SVR/LightGBMgrid;C_3D same4TCNgrid;allinnerselectionandnormalizationtrainingonly. No result-basedpeopleexclusion. Any3Dmatchfailuremarkunmeasurableandreportcommoncohort.',
      'target':'SameprovidedJH-FRATscorewithoutrelabeling. Age65plusheldoutsubgroup. No usingclinicalcomponentsasdeployableinputs.',
      'goals':'Sameengineeringtargets:bothgroupsMAE<=1point,within2>=.8,>=20percentMAEreductionvsround1A;reportconditionalpairedCIanddevelopmentreuse.',
      'limits':'OriginalRGBunavailable;3Dshape/bodyorientation/camerascalearemodelestimates. SourceCSVtwo score/componentinconsistencies to audit,retainprovidedgold. No externalGPU.'})
def iou(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b;inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by));return inter/max(aw*ah+bw*bh-inter,1e-9)
def angle3(a,b,c):
    u=a-b;v=c-b;return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u,v),axis=1),(u*v).sum(1)))
def prepare():
    prot=json.loads((H/'round2-protocol.json').read_text());assert prot['code_sha256']==p.sha(__file__);d=np.load(B/'input.npz');audit=json.loads((H/'input-audit.json').read_text());seq=[];extra=[];matches=[]
    clinical=list(csv.DictReader((p.S/'Extracted/final_gait_feature.csv').open(encoding='utf-8-sig')));labels={r['ID']:r for r in clinical};names=list(clinical[0]);sourcecols=names[1:names.index('age')];assert len(sourcecols)==72
    sourcefeatures=[];eligible=[];failures=[]
    for i,pid in enumerate(d['ids']):
      a2=json.loads((p.S/'Raw/2D'/f'{pid}_2D'/f'{pid}_2D.json').read_text());a3=json.loads((p.S/'Raw/3D'/f'{pid}_3D'/f'{pid}_3D.json').read_text());g2={};g3={}
      for row in a2:g2.setdefault(int(row['image_id'].split('.')[0]),[]).append(row)
      for row in a3:g3.setdefault(int(row['image_id'].split('.')[0]),[]).append(row)
      start=audit['records'][i]['frame_start'];points=[];ious=[]
      failure=None
      for j,frame in enumerate(range(start,start+60)):
        rr=min(g2[frame],key=lambda r:np.max(abs(np.array(r['keypoints']).reshape(26,3)-d['windows'][i,j])));assert np.array_equal(np.array(rr['keypoints']).reshape(26,3),d['windows'][i,j]);cand=max(g3.get(frame,[]),key=lambda r:iou(rr['box'],r['box']),default=None);overlap=iou(rr['box'],cand['box']) if cand is not None else 0
        if overlap<.5:failure={'id':pid,'frame':frame,'reason':'no_same_person_3D_box_IoU_at_least_0.5','maximum_iou':overlap};break
        pts=np.array(cand['pred_xyz_jts'],float);assert pts.shape==(24,3) and np.isfinite(pts).all();points.append(pts);ious.append(overlap)
      sourcefeatures.append([float(labels[pid][k]) for k in sourcecols])
      if failure is not None:
        failures.append(failure);eligible.append(False);seq.append(np.full((72,60),np.nan,np.float32));extra.append([float('nan')]*120);continue
      eligible.append(True);s=savgol_filter(np.array(points),7,2,axis=0);s-=s[:,0,None,:];scale=np.median(np.linalg.norm(s[:,12]-s[:,[7,8]].mean(1),axis=1));assert scale>1e-6;s/=scale;s[:,:,0]*=audit['records'][i]['direction'];F=list(s.std(0).ravel());F.extend(np.sqrt(np.mean(np.sum((np.diff(s,axis=0)*30)**2,axis=2),axis=0)).tolist())
      for a,b,c in [(1,4,7),(2,5,8),(16,1,4),(17,2,5),(16,18,20),(17,19,21)]:
        z=angle3(s[:,a],s[:,b],s[:,c]);F.extend([z.mean(),z.std(),np.quantile(z,.05),np.quantile(z,.95)])
      assert len(F)==120;seq.append(s.reshape(60,-1).T.astype(np.float32));extra.append(F);matches.append({'id':pid,'min_box_iou':min(ious),'frames':60})
      if (i+1)%25==0:print('MATCHED_PROGRESS',i+1,'eligible',sum(eligible),flush=True)
    assert np.isfinite(sourcefeatures).all()
    np.savez_compressed(B/'round2-input.npz',sequence3D=np.array(seq),Xcombined=np.c_[d['X'],np.array(extra)],Xprovider=np.array(sourcefeatures),eligible3D=np.array(eligible),ids=d['ids'],y=d['y'],age=d['age'],fold=d['fold'])
    p.save(H/'round2-input-audit.json',{'registered_people':125,'matched3D_people':int(sum(eligible)),'matched3D_older65':int(sum((d['age']>=65)&np.array(eligible))),'exactmatched_frames':int(sum(eligible))*60,'minimum_box_iou':min(r['min_box_iou'] for r in matches),'matched':matches,'failures':failures,'source_feature_names':sourcecols,'input_sha256':p.sha(B/'round2-input.npz')});print('MATCHED3D',sum(eligible),'PEOPLE',flush=True)
class Temporal3D(l.Temporal):
    def __init__(self,width):
        super().__init__(width);self.enc[0]=nn.Conv1d(72,width,1)
def fit3(S,speed,y,cfg,seed,validation=None,epochs=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);norm=l.normalize_sequence(S,speed);xx,vv=l.tensors(norm,S,speed);ym=float(y.mean());ys=float(max(y.std(),1e-6));yy=torch.tensor((y-ym)/ys,dtype=torch.float32);m=Temporal3D(cfg['width']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=cfg['wd']);lossfn=nn.SmoothL1Loss(beta=.5)
    if validation is not None:vs,vsp,vy=validation;vx,vv2=l.tensors(norm,vs,vsp)
    best=float('inf');bestepoch=0;state=None;stale=0;history=[]
    for epoch in range(1,(epochs or 120)+1):
      m.train()
      for order in np.array_split(rng.permutation(len(y)),max(1,(len(y)+31)//32)):
        xb=xx[order]
        if cfg['noise']:xb=xb+torch.randn_like(xb)*cfg['noise']
        opt.zero_grad();loss=lossfn(m(xb,vv[order]),yy[order]);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
      if validation is not None:
        m.eval()
        with torch.no_grad():pr=np.maximum(0,m(vx,vv2).numpy().astype(float)*ys+ym)
        error=float(abs(pr-vy).mean());history.append(error)
        if error<best-1e-6:best=error;bestepoch=epoch;state=copy.deepcopy(m.state_dict());stale=0
        else:stale+=1
        if stale>=20:break
    if validation is not None:m.load_state_dict(state)
    else:bestepoch=epochs
    return {'kind':'tcn3D','config':cfg,'state':m.state_dict(),'norm':norm,'ymean':ym,'ystd':ys,'ntrain':len(y),'seed':seed,'epochs':int(bestepoch),'history':history}
def pred3(r,S,speed,double=False):
    m=Temporal3D(r['config']['width']);m.load_state_dict(r['state']);m.eval();x,v=l.tensors(r['norm'],S,speed)
    if double:m=m.double();x=x.double();v=v.double()
    with torch.no_grad():pr=m(x,v).numpy().astype(float)
    return np.maximum(0,pr*r['ystd']+r['ymean'])
def train():
    prot=json.loads((H/'round2-protocol.json').read_text());assert prot['code_sha256']==p.sha(__file__);d=np.load(B/'round2-input.npz');base=np.load(B/'input.npz');folder=B/'round2-models';folder.mkdir(exist_ok=True);records=[];start=time.monotonic()
    for f in range(5):
      for arm in ['B_2D3D','C_3D','S_provider']:
        include=np.ones(len(d['y']),bool) if arm=='S_provider' else d['eligible3D'];tr=np.flatnonzero((d['fold']!=f)&include);te=np.flatnonzero((d['fold']==f)&include);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261105).split(tr,np.digitize(d['y'][tr],[6,14])))
        configs=l.TCN if arm=='C_3D' else l.TAB;A=np.empty((len(tr),len(configs)));inner=[];epochs={}
        for k,cfg in enumerate(configs):
          epochs[k]=[]
          for j,(it,iv) in enumerate(splits):
            ti=tr[it];vi=tr[iv]
            if arm=='C_3D':m=fit3(d['sequence3D'][ti],base['X'][ti,0],d['y'][ti],cfg,20261105+f*100+k,validation=(d['sequence3D'][vi],base['X'][vi,0],d['y'][vi]));pr=pred3(m,d['sequence3D'][vi],base['X'][vi,0]);epochs[k].append(m['epochs'])
            else:X=d['Xcombined'] if arm=='B_2D3D' else d['Xprovider'];m=l.fit_tab(X[ti],d['y'][ti],cfg);pr=l.predict_tab(m,X[vi])
            path=folder/f'f{f}-{arm}-k{k}-i{j}.joblib';joblib.dump(m,path,compress=3);A[iv,k]=pr;inner.append({'candidate':k,'inner':j,'train':ti.tolist(),'test':vi.tolist(),'path':str(path),'sha256':p.sha(path)})
        error=abs(A-d['y'][tr,None]).mean(0);k=int(np.argmin(error));cfg=configs[k]
        if arm=='C_3D':m=fit3(d['sequence3D'][tr],base['X'][tr,0],d['y'][tr],cfg,20261105+f*100+k,epochs=max(1,int(np.median(epochs[k]))));pr=pred3(m,d['sequence3D'][te],base['X'][te,0])
        else:X=d['Xcombined'] if arm=='B_2D3D' else d['Xprovider'];m=l.fit_tab(X[tr],d['y'][tr],cfg);pr=l.predict_tab(m,X[te])
        path=folder/f'f{f}-{arm}-final.joblib';joblib.dump(m,path,compress=3);context=folder/f'f{f}-{arm}-context.npz';np.savez_compressed(context,A=A,train=tr,test=te,pred=pr)
        records.append({'fold':f,'arm':arm,'selected':k,'inner_mae':error.tolist(),'inner':inner,'path':str(path),'sha256':p.sha(path),'context':str(context),'context_sha256':p.sha(context)});p.save(H/'round2-receipt.json',{'records':records,'code_sha256':p.sha(__file__),'input_sha256':p.sha(B/'round2-input.npz'),'seconds':time.monotonic()-start});print('ROUND2_DONE',f,arm,round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':{'freeze':freeze,'prepare':prepare,'train':train}[sys.argv[1]]()
