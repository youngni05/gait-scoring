"""Rebuild all frozen-594 goal metrics directly from independently reviewed predictions."""
from pathlib import Path
import json,hashlib,math,statistics,datetime
from collections import defaultdict
import numpy as np
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');G=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q95(values):
    a=sorted(values);position=.95*(len(a)-1);lo=math.floor(position);hi=math.ceil(position);return a[lo]+(a[hi]-a[lo])*(position-lo)
def metrics(rows):
    errors=[abs(r['prediction']-r['actual']) for r in rows];slow=[r for r in rows if r['actual']>=20];m=dict(n=len(rows),mae=statistics.fmean(errors),p95=q95(errors),slow_n=len(slow),slow_mae=statistics.fmean(abs(r['prediction']-r['actual']) for r in slow),slow_bias=statistics.fmean(r['prediction']-r['actual'] for r in slow));m['goal_met']=m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3;m['maximum_normalized_goal_error']=max(m['mae']/.9,m['p95']/3,m['slow_mae']/3);m['bands']={label:dict(n=len(a),mae=statistics.fmean(abs(r['prediction']-r['actual']) for r in a),bias=statistics.fmean(r['prediction']-r['actual'] for r in a),max_error=max(abs(r['prediction']-r['actual']) for r in a)) for label,predicate in [('under12',lambda y:y<12),('12to20',lambda y:12<=y<20),('20plus',lambda y:y>=20)] if(a:=[r for r in rows if predicate(r['actual'])])};return m
def main():
    bridge=read(H/'current-label-bridge.json');assert bridge['status']=='passed' and bridge['current_data_sha256']==sha(GB/'data.npz');data=np.load(GB/'data.npz');truth={int(i):float(y) for i,y in zip(data['ids'],data['y'])};assert len(truth)==594 and sum(y>=20 for y in truth.values())==9
    studies=[(f'round{i:02d}',H/f'round{i:02d}',B/f'round{i:02d}') for i in range(1,13)]+[('local_cnn',G/'local-directml',GB/'local-main'),('native_motion',G/'native-motion',GB/'native-motion'),('norm_aux',G/'norm-aux',GB/'norm-aux'),('manual_controls',G/'manual-controls',GB/'manual-controls')];out={};states=[];failed_diagnostics={}
    for name,small,large in studies:
        vf=small/'verification.json';hf=large/'heldout.json';cf=small/'comparison.json'
        if not(vf.exists() and hf.exists() and cf.exists()):states.append(dict(study=name,status='pending_full_review'));continue
        try:v=read(vf)
        except json.JSONDecodeError:states.append(dict(study=name,status='review_receipt_write_in_progress'));continue
        partial=v.get('status')=='partially_passed_with_auxiliary_logit_failures'
        if v.get('status')!='passed' and not partial:states.append(dict(study=name,status=v.get('status','unknown')));continue
        allowed=set(v['passed_conditions']) if partial else None
        assert v.get('comparison_sha256')==sha(cf)
        if 'heldout_sha256' in v:assert v['heldout_sha256']==sha(hf)
        report=read(cf);groups=defaultdict(list)
        for row in read(hf):
            key=row['condition']+('/'+row['kind'] if 'kind' in row else '');assert row['actual']==truth[int(row['id'])] and math.isfinite(row['prediction']);groups[key].append(row)
        assert set(groups)==set(report)
        for key,rows in groups.items():
            ids=[int(r['id']) for r in rows];assert len(ids)==594 and len(set(ids))==594 and set(ids)==set(truth);m=metrics(rows);assert m['slow_n']==9
            for field in ['mae','p95','slow_mae','slow_bias']:
                if field in report[key]:assert math.isclose(m[field],report[key][field],abs_tol=1e-12,rel_tol=1e-12)
            assert m['goal_met']==report[key]['goal_met'];m['study']=name;m['condition']=key;m['source_heldout']=str(hf);m['source_heldout_sha256']=sha(hf);m['source_verification']=str(vf);m['source_verification_sha256']=sha(vf)
            if partial and key not in allowed:
                m['eligible_for_goal_claim']=False;m['failure_reason']=v['condition_status'][key];failed_diagnostics[name+'/'+key]=m
            else:out[name+'/'+key]=m
        states.append(dict(study=name,status='partial_source_review_failed_condition_not_eligible' if partial else 'passed_source_review_and_independent_goal_metric_rebuild',conditions=len(groups),passed_conditions=sorted(allowed) if partial else list(groups),failed_conditions=v.get('failed_conditions',[]),source_verification_sha256=sha(vf)))
    pending=[r['study'] for r in states if r['status']=='pending_full_review'];hits=[k for k,v in out.items() if v['goal_met']];best_mae=min(out,key=lambda k:out[k]['mae']);closest=min(out,key=lambda k:out[k]['maximum_normalized_goal_error']);pareto=[]
    for key,value in out.items():
        a=[value[x] for x in ['mae','p95','slow_mae']]
        if not any(all(b<=x for b,x in zip([other[t] for t in ['mae','p95','slow_mae']],a)) and any(b<x for b,x in zip([other[t] for t in ['mae','p95','slow_mae']],a)) for k,other in out.items() if k!=key):pareto.append(key)
    payload=dict(status='independent_goal_metrics_rebuilt',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),people=594,slow20plus=9,goal=dict(mae_max=.9,p95_max=3,slow_mae_max=3),studies=states,verified_conditions=len(out),best_MAE_condition=best_mae,closest_three_goal_condition=closest,pareto_conditions=pareto,numeric_goal_hits=hits,pending_studies=pending,failed_condition_diagnostics=failed_diagnostics,completion_proven=False,limits='Numericgateonly,nottaskcompletion. Pendingmodels/reviews/reportsmustfinish;anyhitrequiresfreshsource-specificfullmodel/person/epoch/scaler/recoveryauditandpaireduncertaintybeforegoalcompletion. Reuseddevelopmentfolds,notexternalclinicalaccuracy. Stdlibfmeanandmanualsortedlinear95percentile independentlyrebuildmetrics;all594and9slowrequired,nomissingrowsoroutcomeexclusions. Failedconditionsrecordedbutneverusedasgoalhits;otherindependentconditionsmustfullypassallsamegatesontheentirecohort. Source-specificuncertaintyfilesretainconditionalpairedintervals.',conditions=out,code_sha256=sha(Path(__file__)));(H/'scoreboard.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({k:payload[k] for k in ['verified_conditions','best_MAE_condition','closest_three_goal_condition','numeric_goal_hits','pending_studies']},indent=2))
if __name__=='__main__':main()
