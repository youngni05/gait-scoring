from study import *

def main():
    spec=importlib.util.spec_from_file_location('prior_assess',HERE.parent/'20261031-opencap-leave283/assess.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    protocol=json.loads((HERE/'protocol.json').read_text());summary={}
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort;assert json.loads((out/'completion.json').read_text())['complete']
        p=pd.read_csv(out/'predictions.csv');p['scenario']=cohort;p['family']='Ridge'
        part,persons,runs,metrics=module.aggregate(p)
        part.to_csv(out/'partition-person.csv',index=False);persons.to_csv(out/'person-results.csv',index=False)
        persons[persons.target>=20].to_csv(out/'long-people.csv',index=False);runs.to_csv(out/'partition-metrics.csv',index=False)
        pairs=[(f'{kind}{th}_{b}',f'{base}_{b}') for b in [10,20,30] for th in ['10','15'] for kind in ['Add','Move'] for base in ['Speed',f'Learned{th}']]
        pairs += [(f'Move{th}_{b}',f'Add{th}_{b}') for b in [10,20,30] for th in ['10','15']]
        contrasts=[module.contrast(persons,runs,(cohort,'Ridge',a),(cohort,'Ridge',b),band) for a,b in pairs for band in ['all','ge20','lt20']]
        summary[cohort]={'metrics':metrics,'contrasts':contrasts};save(cohort+'/summary.json',summary[cohort]);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        print(cohort,pd.DataFrame(metrics)[['view','band','people','MAE','underprediction','overprediction']].to_string(index=False),flush=True)
        for c in contrasts:
            if c['band']=='ge20' and c['a'][2]=='Add10_20':
                print('LONG_COMPARISON',c['a'][2],c['b'][2],json.dumps({k:{q:c[k][q] for q in ['difference','conditional_bootstrap95','better_partitions','people_better']} for k in ['MAE','underprediction']}),flush=True)
    save('summary.json',summary)
if __name__=='__main__':main()
