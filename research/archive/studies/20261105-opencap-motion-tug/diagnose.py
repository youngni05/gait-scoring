"""Describe >1s errors without retuning on the outer-fold outcomes."""
import csv
import numpy as np
import prepare as p
H=p.H;B=p.B

def main():
    result=p.read(H/'results.json');v=p.read(H/'verification.json');assert v['status']=='passed' and p.sha(H/'results.json')==v['results_sha256']
    d=np.load(B/'input.npz');pr=np.load(B/'verified-predictions.npz');audit=p.read(H/'input-audit.json');receipt=p.read(H/'fit-receipt.json')
    leg=np.array([r['leg_length_m'] for r in audit['records']]);net=d['sequence'][:,-2,-1].astype(float)*leg/3.;ratio=d['X'][:,0]/np.maximum(net,1e-6)
    intervals={};errors=[]
    for arm in ['B','C','D','R']:
        radius=np.empty(297)
        for row in receipt['records']:
            tr=np.array(row['train']);te=np.array(row['test']);ctx=np.load(row['context']);r=float(np.quantile(abs(ctx['inner_'+arm]-d['y'][tr]),.9,method='higher'));radius[te]=r
        intervals[arm]={name:{'n':int(sum(mask)),'median_radius_seconds':float(np.median(radius[mask])),'empirical_coverage':float(np.mean(abs(pr[arm][mask]-d['y'][mask])<=radius[mask]))} for name,mask in [('all',np.ones(297,bool)),('age65plus',d['age']>=65),('TUG20plus',d['y']>=20)]}
    for i,pid in enumerate(d['ids']):
        for arm in ['A','B','C','D','R','E']:
            error=float(pr[arm][i]-d['y'][i])
            if abs(error)>1:errors.append({'id':str(pid),'arm':arm,'actual_TUG':float(d['y'][i]),'predicted_TUG':float(pr[arm][i]),'error_seconds':error,'age':float(d['age'][i]),'speed_mps':float(d['X'][i,0]),'net_speed_mps':float(net[i]),'meanabs_to_net_ratio':float(ratio[i])})
    with (H/'errors-over1second.csv').open('w',newline='',encoding='utf8') as file:writer=csv.DictWriter(file,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
    cases=[]
    for i in np.flatnonzero(d['y']>=20):
        cases.append({'id':str(d['ids'][i]),'TUG_seconds':float(d['y'][i]),'age':float(d['age'][i]),'diagnosis':str(d['diagnosis'][i]),'speed_mps':float(d['X'][i,0]),'net_speed_mps':float(net[i]),'meanabs_to_net_ratio':float(ratio[i]),'illustrative6m_over_speed_seconds':float(6/max(d['X'][i,0],.1)),**{a:float(pr[a][i]) for a in ['A','B','C','D','R','E']}})
    counts={a:int(sum(abs(pr[a]-d['y'])>1)) for a in ['A','B','C','D','R','E']}
    flagged=np.flatnonzero(ratio>1.1);flagged_cases=[]
    for i in flagged:
        flagged_cases.append({'id':str(d['ids'][i]),'age':float(d['age'][i]),'actual_TUG':float(d['y'][i]),'speed_meanabs_mps':float(d['X'][i,0]),'net_speed_mps':float(net[i]),'backward_steps_outof90':int(sum(np.diff(d['sequence'][i,-2])<0)),**{a:float(pr[a][i]) for a in ['A','B','C','D','E']}})
    keep=ratio<=1.1
    sensitivity={a:float(abs(pr[a][keep]-d['y'][keep]).mean()) for a in ['A','B','C','D','E']}
    p.save(H/'diagnosis.json',{'units':'seconds','errors_over1second_per_arm':counts,'slow9_profiles':cases,
      'speed_definition_audit':{'meanabs_dividedby_netspeed_median':float(np.median(ratio)),'p95':float(np.quantile(ratio,.95)),'maximum':float(ratio.max()),'ratio_over1point1_people':int(sum(ratio>1.1)),'slow9_maxratio':float(ratio[d['y']>=20].max()),'direction_reversal_flagged_cases':flagged_cases,'same_frozen_prediction_MAE_without_flagged1_descriptive_only':sensitivity,'primary_people_or_inputs_changed':False,'limits':'Netdisplacementisnotindependentgroundtruth. Directionreversalcanbeactualturnorreconstructionjitter;RGB/phaseannotationsunavailable. Ratio1.1isposthocdiagnostic,notnewtrainingqualitygate. Calibrated3Dcoordinateerrorremainsunknown.'},
      'empirical_90percent_inner_residual_intervals':intervals,'interval_limits':'Train-innerOOFresidualquantile,notformalcoverageguarantee/clinicalMDC;crossfitmodelsdifferfromfinalfit. Eexcludedbecausemeta-in-sampleresidualwouldbeoptimistic.',
      'confirmed':'Inputunits,personvisitlabels,window/sourcehashes,all64featureformulasandfrozenmodelsreplaypassed. Largeerroraloneisnotproofofsoftwarebug.',
      'remaining_hypotheses':['3secondstraightwalkcannotdirectlyobservechair-rise/turn/sittingperformance','Only9peoplehaveTUG>=20;modelscompressupperrange','Videoderived3Dandidealsideprojectionmayretainmeasurementerror;realphoneerrornotmeasured','Neuromuscularcohortand34olderpeopledonotrepresentallcommunityelderly'],
      'anchor_limit':'6/v isonlyanillustrativewalkinganchor. DifferencefromTUGisnotameasuredturnorstandduration.',
      'no_new_training_from_diagnostic':True,'promising_declared_by_frozen_rule':result['promising_for_four_repeat_sensitivity']})
    print('DIAGNOSED',counts,'RATIO_MAX',float(ratio.max()),flush=True)

if __name__=='__main__':main()
