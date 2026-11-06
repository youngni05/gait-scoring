from common import *
from collections import Counter
audit=json.loads((HERE/'turn-audit.json').read_text());m=pd.read_csv(PREV/'manifest.csv');clinical=pd.read_csv(SOURCE/'raw/participant_info.csv').set_index(['subid','visit']);selected=json.loads((HERE/'selected-files.json').read_text())
rows=[]
for r in audit:
 q=dict(r);q['target']=float(clinical.loc[(r['id'],r['visit']),'tft_tug'])
 if r['detection']=='passed':
  a=read(HERE/'raw'/r['file']);q['recording_seconds']=float(a['time'][-1]-a['time'][0]);q['recording_shorter_than_label_by_gt1s']=q['recording_seconds']<q['target']-1
 rows.append(q)
save('linkage-audit.json',rows)
cohorts={};long=[]
for name in ['event_matched','fixed6','event_flagged']:
 accepted=json.loads((HERE/name/'manifest.json').read_text());excluded=json.loads((HERE/name/'excluded.json').read_text());ar={r['id']:r for r in accepted};er={r['id']:r for r in excluded}
 cohorts[name]={'accepted_people':len(accepted),'excluded_people':len(excluded),'reasons':dict(Counter(reason for r in excluded for reason in r['reasons'])),'line_cases':[{'id':r['id'],'visit':r['visit'],'target':r['target']} for r in accepted if r['task']=='line'],'recording_shorter_than_label_by_gt1s':[r['id'] for r in accepted if next(x for x in rows if x['file']==r['turn_file'])['recording_shorter_than_label_by_gt1s']]}
 for old in m[m.target>=20].to_dict('records'):
  rr={'cohort':name,'id':old['id'],'visit':old['visit'],'target':old['target'],'included':old['id'] in ar,'same_visit_files':sum(identity(x['name'])[:2]==(old['id'],old['visit']) for x in selected)}
  if old['id'] in ar:rr.update({'turn_seconds':ar[old['id']]['turn_observed_seconds'],'task':ar[old['id']]['task'],'walk_qc':ar[old['id']]['walk_qc'],'turn_qc':ar[old['id']]['turn_qc']})
  else:rr['exclusion']=er[old['id']]['reasons'];rr['detection_statuses']=[r['detection'] for r in audit if r['id']==old['id']]
  long.append(rr)
save('coverage.json',{'prior_people':len(m),'same_visit_tug_people':len(set(r['id'] for r in audit)),'downloaded_files':len(audit),'detection_statuses':dict(Counter(r['detection'] for r in audit)),'cohorts':cohorts,'source_label_scope':'exact subject/visit key and prior target match; individual timed trial mapping not provided'})
save('long-people-coverage.json',long)
print(json.dumps(json.loads((HERE/'coverage.json').read_text())),flush=True)
