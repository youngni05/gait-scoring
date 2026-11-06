from study import *

def main():
    spec=importlib.util.spec_from_file_location('prior_assess',HERE.parent/'20261031-opencap-leave283/assess.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    protocol=json.loads((HERE/'protocol.json').read_text());summary={};case_rows=[];diagnostics=[]
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort;p=pd.read_csv(out/'predictions.csv');p['scenario']=cohort;p['family']='Ridge'
        part,persons,runs,metrics=module.aggregate(p)
        part.to_csv(out/'partition-person.csv',index=False);persons.to_csv(out/'person-results.csv',index=False)
        persons[persons.target>=20].to_csv(out/'long-people.csv',index=False);runs.to_csv(out/'partition-metrics.csv',index=False)
        pairs=[(f'{a}_{b}',f'{c}_{b}') for b in [10,20,30] for a,c in [('Pooled','Baseline'),('Directional','Baseline'),('Directional','Pooled')]]
        contrasts=[module.contrast(persons,runs,(cohort,'Ridge',a),(cohort,'Ridge',c),band) for a,c in pairs for band in ['all','ge20','lt20']]
        summary[cohort]={'metrics':metrics,'contrasts':contrasts};save(cohort+'/summary.json',summary[cohort]);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        for (view,pid),g in p[p.id.isin(['sub-397','sub-283','sub-288'])].groupby(['view','id']):
            error=g.prediction-g.target
            case_rows.append({'cohort':cohort,'view':view,'id':pid,'target':float(g.target.iloc[0]),'prediction':float(g.prediction.mean()),'sd':float(g.prediction.std(ddof=0)),'minimum':float(g.prediction.min()),'maximum':float(g.prediction.max()),'MAE':float(abs(error).mean()),'underprediction':float(np.maximum(-error,0).mean())})
        for view,g in persons.groupby('view'):
            for name,sub in [('long_without397',g[(g.target>=20)&(g.id!='sub-397')]),('common176',g[g.id!='sub-283']),('common8long',g[(g.target>=20)&(g.id!='sub-283')])]:
                diagnostics.append({'cohort':cohort,'view':view,'subset':name,'people':len(sub),**{k:float(sub[k].mean()) for k in ['MAE','underprediction','overprediction']}})
        print(cohort,pd.DataFrame(metrics).query("band in ['all','ge20']")[['view','band','MAE','underprediction','overprediction']].to_string(index=False),flush=True)
    save('summary.json',summary);pd.DataFrame(case_rows).to_csv(HERE/'case-results.csv',index=False);pd.DataFrame(diagnostics).to_csv(HERE/'diagnostic-subsets.csv',index=False)
if __name__=='__main__':main()
