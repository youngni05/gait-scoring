"""Explain >1point trigger without changing gold labels or dropping difficult people."""
from pathlib import Path
import csv,json,collections
import numpy as np
import prepare as p
import review as r
H=p.H;B=p.B
def main():
    d=np.load(B/'input.npz');pr=np.load(B/'verified-predictions.npz');raw=list(csv.DictReader((p.S/'Extracted/final_gait_feature.csv').open(encoding='utf-8-sig')));meta={z['ID']:z for z in raw};cols=['ageGroup','fallsHistory','elimination_bowel_urine','medications','patient_care_equipment','mobility','cognition'];components=np.array([[float(meta[pid][c]) for c in cols] for pid in d['ids']]);sums=components.sum(1);mismatch=np.flatnonzero(sums!=d['y']);bad=[]
    for i in mismatch:bad.append({'id':str(d['ids'][i]),'provided_score':float(d['y'][i]),'component_sum':float(sums[i]),'components':dict(zip(cols,components[i].tolist()))})
    meanpoints=components.mean(0);part={'columns':cols,'mean_points':dict(zip(cols,meanpoints.tolist())),'share_of_component_sum':dict(zip(cols,(meanpoints/meanpoints.sum()).tolist())),'mismatch_count':len(mismatch),'mismatches':bad,'primary_gold_changed':False}
    byband={};cases=[]
    for g,mask in [('all',np.ones(len(d['y']),bool)),('age65plus',d['age']>=65)]:
      byband[g]={}
      for arm in ['A','B','C','D']:
        yy=d['y'][mask];err=pr[arm][mask]-yy;byband[g][arm]=[{'low':lo,'high':hi,'n':int(sum((yy>=lo)&(yy<hi))),'mae':float(abs(err[(yy>=lo)&(yy<hi)]).mean()) if sum((yy>=lo)&(yy<hi)) else None,'bias':float(err[(yy>=lo)&(yy<hi)].mean()) if sum((yy>=lo)&(yy<hi)) else None,'MAEcontribution':float(abs(err[(yy>=lo)&(yy<hi)]).sum()/len(yy))} for lo,hi in [(0,6),(6,14),(14,999)]]
    for i in np.argsort(abs(pr['C']-d['y']))[-12:][::-1]:
        cases.append({'id':str(d['ids'][i]),'age':float(d['age'][i]),'score':float(d['y'][i]),'normalized_speed':float(d['X'][i,0]),'A':float(pr['A'][i]),'B':float(pr['B'][i]),'C':float(pr['C'][i]),'age_only':float(pr['age'][i]),'components':dict(zip(cols,components[i].tolist()))})
    # Descriptive only: nearest visibly similar trajectories after per-joint z scaling.
    x=np.c_[d['sequence'].mean(2),d['sequence'].std(2),d['X'][:,0]];xx=(x-x.mean(0))/np.maximum(x.std(0),1e-6);distance=np.sqrt(((xx[:,None]-xx[None,:])**2).mean(2));np.fill_diagonal(distance,np.inf)
    pairs=[]
    for i in range(len(x)):
        j=int(np.argmin(distance[i]));pairs.append({'id':str(d['ids'][i]),'nearest':str(d['ids'][j]),'normalized_descriptor_distance':float(distance[i,j]),'score_difference':float(abs(d['y'][i]-d['y'][j]))})
    topneighbors=sorted(pairs,key=lambda z:z['score_difference'],reverse=True)[:10]
    sensitivity={}
    mask=np.ones(125,bool);mask[mismatch]=False
    for arm in ['A','B','C','D']:sensitivity[arm]={'same_frozen_predictions_excluding2_inconsistent_rows_MAE':float(abs(pr[arm][mask]-d['y'][mask]).mean()),'not_a_retrained_model':True}
    p.save(H/'diagnosis.json',{'trigger':'AllprimaryarmsMAE>1point;engineeringgoalsnotmet','label_composition':part,'error_bands':byband,'large_error_cases':cases,'nearest_pose_descriptors':topneighbors,'label_consistency_sensitivity':sensitivity,
      'verified_not_detected':['unitcopyorinputformulaerror','samepersontrain/testoverlap','testsetnormalization','outertestepochselection','nonfiniteinput'],'remaining_hypotheses':['Labelincludesnonvisualriskfactorsthatareonlyindirectlyrelatedtogait','2secprojectedmotionlosesdepthandotherTUGphases','smallcohortand5upper-tailpeoplecausepredictioncompression','trackselectionandstraight-gaitphasearenotindependentlyannotated'],
      'scope':'Associationaldiagnostics,noproofclinicalscoreunpredictability. Keepall125originalgoldlabels;123-rowcheckdescriptiveonly. No scorecomponentordeploymentageinputadded.'})
    print('DIAGNOSED',len(bad),'LABEL_COMPONENT_MISMATCHES','mobilityfraction',part['share_of_component_sum']['mobility'],flush=True)
if __name__=='__main__':main()
