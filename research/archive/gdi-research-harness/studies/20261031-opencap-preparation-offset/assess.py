from study import *

def main():
    helper=module('aggregate_prior',HERE.parent/'20261031-opencap-leave283/assess.py')
    protocol=json.loads((HERE/'protocol.json').read_text());summary={};stability=[];coverage=[]
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort;p=pd.read_csv(out/'predictions.csv');p['scenario']=cohort;p['family']='Ridge'
        part,persons,runs,metrics=helper.aggregate(p)
        for name,frame in [('partition-person',part),('person-results',persons),('partition-metrics',runs),('metrics',pd.DataFrame(metrics)),('long-people',persons[persons.target>=20])]:frame.to_csv(out/f'{name}.csv',index=False)
        pairs=[(f'{a}_{b}',f'{z}_{b}') for b in [10,20,30] for a,z in [('Prep','Gait'),('Core','Gait'),('Reallocated','Gait'),('Prep','Reallocated'),('Core','Reallocated'),('Prep','Core'),('LearnPrep','Prep')]]
        contrasts=[helper.contrast(persons,runs,(cohort,'Ridge',a),(cohort,'Ridge',z),band) for a,z in pairs for band in ['all','ge20','lt20']]
        summary[cohort]={'metrics':metrics,'contrasts':contrasts};save(cohort+'/summary.json',summary[cohort])
        for (view,id),g in p[p.target>=20].groupby(['view','id']):
            q=g.prediction.to_numpy();e=q-g.target.to_numpy()
            stability.append({'cohort':cohort,'view':view,'id':id,'target':float(g.target.iloc[0]),'mean':float(q.mean()),'sd':float(q.std()),'min':float(q.min()),'max':float(q.max()),'MAE':float(abs(e).mean()),'underprediction':float(np.maximum(-e,0).mean()),'predictions':q.tolist()})
        # Evaluate the main contrast also without the two highlighted people, without retraining.
        sensitivities=[]
        for omitted in [[],['sub-283'],['sub-397'],['sub-283','sub-397']]:
            pp=persons[(persons.target>=20)&~persons.id.isin(omitted)]
            for b in [10,20,30]:
                a=pp[pp.view==f'Prep_{b}'].set_index('id');z=pp[pp.view==f'Gait_{b}'].set_index('id').loc[a.index]
                sensitivities.append({'omit_evaluation_only':omitted,'budget':b,'people':len(a),**{k:float((a[k]-z[k]).mean()) for k in ['MAE','underprediction','overprediction']}})
        save(cohort+'/case-exclusion-sensitivity.json',sensitivities)
        print(cohort,pd.DataFrame(metrics).query("band=='ge20'")[['view','people','MAE','underprediction','overprediction']].to_string(index=False),flush=True)
    for m in json.loads((HERE/'include283/manifest.json').read_text()):
        ph=m['phase'];full=ph['extended'][1]-ph['extended'][0];core=ph['core'][1]-ph['core'][0]
        for b in [10,20,30]:
            r=m['comparison_budgets'][str(b)]
            coverage.append({'id':m['id'],'target':m['target'],'budget':b,'full_prep_rise':full,'full_core':core,'full_preparatory':full-core,
              'outside_old_window':r['full_preparation_rise_outside_original'],'observed_prep_rise':r['rise_observed'],'observed_core':r['core_observed'],'observed_preparatory':r['preparation_observed'],
              'still_unobserved':r['unobserved_rise'],'old_gait_stop':r['original']['delay_features']['10'][2],'new_gait_stop':r['reallocated_gait']['delay_features']['10'][2],
              'actual_observed':r['total_observed'],'elapsed_span_with_gaps':r['elapsed_span_with_gaps']})
    save('summary.json',summary);save('stability.json',stability);pd.DataFrame(stability).drop(columns=['predictions']).to_csv(HERE/'stability.csv',index=False)
    save('coverage.json',coverage);pd.DataFrame(coverage).to_csv(HERE/'coverage.csv',index=False)
    print(pd.DataFrame(stability).query("id in ['sub-283','sub-397'] and view in ['Gait_20','Core_20','Prep_20']").drop(columns=['predictions']).to_string(index=False),flush=True)
if __name__=='__main__':main()
