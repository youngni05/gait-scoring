from pathlib import Path
import json,sys,importlib.util,re
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261031-opencap-directional-pace'
spec=importlib.util.spec_from_file_location('pace_source',PRIOR/'study.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
np,pd,sha,read,joblib=source.np,source.pd,source.sha,source.read,source.joblib
LABELS=HERE.parent/'20261030-older44-sppb-timeseries/candidate-audit/nmd_opencap_participant_info.csv'
CHAIR=HERE.parent/'20261031-opencap-chair'
def save(n,v):
    p=HERE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def main():
    labels=pd.read_csv(LABELS);assert not labels.duplicated(['subid','visit']).any();lookup=labels.set_index(['subid','visit'])
    cohort=PRIOR/'include283';d=np.load(cohort/'input.npz');rows=json.loads((cohort/'manifest.json').read_text())
    phases={r['id']:r for r in json.loads((CHAIR/'transition-audit.json').read_text())}
    scouts={r['id']:r for r in json.loads((source.long.HERE/'scout.json').read_text())}
    persons=pd.read_csv(cohort/'person-results.csv');exclusive=pd.read_csv(PRIOR/'exclude283/person-results.csv')
    details=[]
    for i,m in enumerate(rows):
        pid=m['id'];lab=lookup.loc[(pid,m['visit'])];assert float(lab.tft_tug)==m['target']
        filepid,visit=re.search(r'(sub-\d+)/visit-(\d+)',m['turn_file']).groups();assert filepid==pid and int(visit)==m['visit']
        p=phases[pid];a=read(source.long.TURN/'raw'/m['turn_file']);assert sha(source.long.TURN/'raw'/m['turn_file'])==m['turn_sha256']
        s=scouts.get(pid)
        if s is None:
            pre,post,_=source.long.old.eligible(a,m);s={'pre':pre,'post':post}
        r={'id':pid,'visit':m['visit'],'target':m['target'],'diag':lab['diag'],'age':float(lab.age),'sex':lab.sex,'clinical_5xsts':float(lab.tft_5xsts) if pd.notna(lab.tft_5xsts) else None,
           'turn_seconds':m['turn_observed_seconds'],'outbound_speed':m['pace']['20']['outbound']['speed'],'return_speed':m['pace']['20']['return']['speed'],
           'gait_stop20':m['budgets']['20']['delay_features']['10'][2], 'observed20':m['budgets']['20']['observed_seconds'],'observed30':m['budgets']['30']['observed_seconds'],
           'valid_gait20':m['budgets']['20']['valid_nonturn_seconds'],'seated_removed20':m['budgets']['20']['seated_removed_seconds'],'tracking_removed20':m['budgets']['20']['tracking_removed_seconds'],
           'turn_max_horizontal_speed':m['turn_max_speed'],'pre':s['pre'],'post':s['post'],'phases':p,'source_file':m['turn_file'],'source_sha256':m['turn_sha256']}
        for kind in ['Baseline','Pooled','Directional']:
            use=persons if pid=='sub-283' else exclusive
            q=use[(use.id==pid)&(use.view==kind+'_20')].iloc[0]
            r[kind+'_prediction']=float(q.prediction);r[kind+'_under']=float(q.underprediction);r[kind+'_MAE']=float(q.MAE)
        r['large_under']=bool(m['target']>=20 and r['Baseline_under']>=5 and pid!='sub-283')
        if all('window' in p[k] for k in ['rise','sit']):
            rise,sit=p['rise']['window'],p['sit']['window'];span=sit[1]-rise[0]
            def intersect(x,y):return max(0,min(x[1],y[1])-max(x[0],y[0]))
            between=[rise[1],sit[0]];walking_intervals=[q for q in [s['pre'],[m['turn_start'],m['turn_end']],s['post']] if q]
            represented=sum(intersect(q,between) for q in walking_intervals)
            r.update(core_rise_to_sit_span=span,clinical_minus_core_span=m['target']-span,rise_seconds=rise[1]-rise[0],sit_seconds=sit[1]-sit[0],
              between_rise_end_and_sit_start=between[1]-between[0],full_eligible_gait_turn_seconds=represented,
              between_outside_gait_mask=between[1]-between[0]-represented,
              after_rise_before_eligible=max(0,s['pre'][0]-rise[1]) if s['pre'] else None,
              after_eligible_before_sit=max(0,sit[0]-s['post'][1]) if s['post'] else None,
              phase_quality=bool(p['rise']['strict_quality'] and p['sit']['strict_quality']))
        details.append(r)
    save('person-audit.json',details)
    frame=pd.json_normalize(details);frame.to_csv(HERE/'person-audit.csv',index=False)
    frame[frame.target>=20].to_csv(HERE/'long-audit.csv',index=False)
    summary={}
    for name,mask in [('all176',frame.id!='sub-283'),('short168',frame.target<20),('long8',(frame.target>=20)&(frame.id!='sub-283')),('large_under4',frame.large_under),('otherlong4',(frame.target>=20)&~frame.large_under&(frame.id!='sub-283'))]:
        f=frame[mask];summary[name]={'people':len(f)}
        for k in ['outbound_speed','return_speed','turn_seconds','gait_stop20','observed20','between_outside_gait_mask','clinical_minus_core_span','rise_seconds','sit_seconds']:
            z=f[k].dropna();summary[name][k]={'n':len(z),'median':float(z.median()) if len(z) else None,'range':[float(z.min()),float(z.max())] if len(z) else None}
    valid=labels[pd.to_numeric(labels.tft_tug,errors='coerce')>0];tail=valid[valid.tft_tug>=20];known=set(d['ids']);outside=tail[~tail.subid.isin(known)]
    save('clinical-coverage.json',{'rows':len(labels),'unique_people':labels.subid.nunique(),'positive_tug_rows':len(valid),'positive_tug_people':valid.subid.nunique(),
      'long_visits':len(tail),'long_people':tail.subid.nunique(),'long_people_outside177':sorted(set(outside.subid)),
      'outside_long_visits':outside[['subid','visit','tft_tug','diag']].to_dict('records'),
      'selected_long':frame[frame.target>=20][['id','visit','target','diag']].to_dict('records'),
      'selected_people_other_long_visits':tail[tail.subid.isin(known)][['subid','visit','tft_tug']].to_dict('records')})
    save('group-summary.json',summary)
    # Descriptive training-support diagnosis: no new fitted predictions or clinical labels as inputs.
    trainrows=[];neighbors=[];foldcounts=[]
    for cohortname in ['exclude283','include283']:
        out=PRIOR/cohortname;data=np.load(out/'input.npz');y=data['y'];ids=data['ids'];plans={(r['repeat'],f['fold']):f for r in json.loads((out/'splits.json').read_text()) for f in r['splits']}
        for sel in json.loads((out/'selections.json').read_text()):
            if sel['view'] not in ['Baseline_20','Pooled_20']:continue
            f=plans[sel['repeat'],sel['fold']];tr=np.array(f['train']);te=np.array(f['test']);x=data[sel['view']][:,:-1];offset=data[sel['view']][:,-1]
            assert sha(PRIOR/sel['file'])==sel['sha256'];sc,model=joblib.load(PRIOR/sel['file']);z=sc.transform(x);pred=np.maximum(model.predict(z),0)+offset
            foldcounts.append({'cohort':cohortname,'repeat':sel['repeat'],'fold':sel['fold'],'view':sel['view'],'train_people':len(tr),'train_long':int(sum(y[tr]>=20)),'alpha':sel['alpha']})
            for k in np.flatnonzero(y>=20):
                trainrows.append({'cohort':cohortname,'view':sel['view'],'repeat':sel['repeat'],'fold':sel['fold'],'id':str(ids[k]),'target':float(y[k]),'role':'training' if k in tr else 'heldout','prediction':float(pred[k]),'under':float(max(y[k]-pred[k],0)),'MAE':float(abs(y[k]-pred[k]))})
                if k not in te:continue
                distances=np.linalg.norm(z[tr]-z[k],axis=1);near=tr[np.argsort(distances)[:5]]
                neighbors.append({'cohort':cohortname,'view':sel['view'],'repeat':sel['repeat'],'id':str(ids[k]),'neighbor_ids':ids[near].tolist(),'neighbor_targets':y[near].tolist(),'long_neighbors':int(sum(y[near]>=20)),'target_mean':float(y[near].mean()),'nearest_distance':float(min(distances)),'target_outside_feature_range':int(sum((x[k]<x[tr].min(0))|(x[k]>x[tr].max(0))))})
    pd.DataFrame(trainrows).to_csv(HERE/'train-vs-heldout.csv',index=False);save('neighbors.json',neighbors);save('fold-counts.json',foldcounts)
    n=pd.DataFrame(neighbors);n.groupby(['cohort','view','id'])[['long_neighbors','target_mean','nearest_distance','target_outside_feature_range']].mean().reset_index().to_csv(HERE/'neighbor-summary.csv',index=False)
    pd.DataFrame(trainrows).groupby(['cohort','view','id','role'])[['prediction','under','MAE']].mean().reset_index().to_csv(HERE/'train-support-summary.csv',index=False)
    files=[LABELS,CHAIR/'transition-audit.json',source.long.HERE/'scout.json']+[PRIOR/c/n for c in ['include283','exclude283'] for n in ['input.npz','manifest.json','splits.json','person-results.csv','selections.json']]
    save('sources.json',{str(p.relative_to(HERE.parent)):sha(p) for p in files})
    print(frame[frame.target>=20][['id','target','diag','Baseline_prediction','between_outside_gait_mask','clinical_minus_core_span','phase_quality','tracking_removed20']].to_string(index=False),flush=True)
    print(json.dumps(json.loads((HERE/'clinical-coverage.json').read_text())),flush=True)
if __name__=='__main__':main()
