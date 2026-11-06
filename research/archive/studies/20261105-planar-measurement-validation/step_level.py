"""Exploratory matched-step description; preserves every primary window/estimate.

Only adjacent opposite-foot events matched in both streams count. Report conditional
coverage: this does not estimate accuracy for unmatched or rejected steps.
"""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import measurement as m
from review import cluster_stats

def main():
    verification=m.read(m.H/'verification.json')
    assert verification['status']=='passed'
    assert verification['measurements_sha256']==m.sha(m.H/'measurements.json')
    rows=m.read(m.H/'measurements.json')['records'];out=[]
    for row in rows:
        if row['group']!='primary2s' or row['camera']!='PG1' or not row.get('foot_quality'):continue
        p=row['predicted_events'];t=row['reference_events'];mapping={}
        for side in [0,1]:
            pp=[i for i,e in enumerate(p) if e['side']==side];tt=[i for i,e in enumerate(t) if e['side']==side]
            if pp and tt:
                cost=np.array([[abs(p[i]['time']-t[j]['time']) for j in tt] for i in pp]);a,b=linear_sum_assignment(cost)
                for i,j in zip(a,b):
                    if cost[i,j]<=.15:mapping[pp[i]]=tt[j]
        for i in range(1,len(p)):
            if i not in mapping or i-1 not in mapping or mapping[i]!=mapping[i-1]+1 or p[i]['side']==p[i-1]['side']:continue
            ps=next((s for s in row['predicted_steps'] if s['side']==p[i]['side'] and abs(s['time']-p[i]['time'])<1e-9),None)
            ts=next((s for s in row['reference_steps'] if s['side']==t[mapping[i]]['side'] and abs(s['time']-t[mapping[i]]['time'])<1e-9),None)
            if ps and ts:out.append({'subject':row['subject'],'key':row['key'],'prediction':ps,'reference':ts})
    result={'analysis':'exploratory_matched_adjacent_steps_not_all_step_accuracy','rule':'same-foot Hungarian assignment <=0.15s; consecutive matched opposite-foot events must also be consecutive in reference; original accepted step definitions unchanged','people':len(set(x['subject'] for x in out)),'matched_step_pairs':len(out),'registered_primary_people':14,'measures':{field:cluster_stats([(x['subject'],x['reference'][field],x['prediction'][field]) for x in out]) for field in ['step_time','step_length']},'pooled_step_MAE':{field:float(np.mean([abs(x['prediction'][field]-x['reference'][field]) for x in out])) for field in ['step_time','step_length']},'pairs':out,'measurements_sha256':m.sha(m.H/'measurements.json'),'verification_sha256':m.sha(m.H/'verification.json'),'code_sha256':m.sha(__file__)}
    m.save(m.H/'matched-step-results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))

if __name__=='__main__':main()
