from study import *

def main():
    assert json.loads((HERE/'completion.json').read_text())['complete']
    spec=importlib.util.spec_from_file_location('prior_assess',HERE.parent/'20261031-opencap-leave283/assess.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=pd.read_csv(HERE/'predictions.csv');p['scenario']='Excluded';p['family']='Ridge'
    part,persons,runs,metrics=module.aggregate(p)
    part.to_csv(HERE/'partition-person.csv',index=False);persons.to_csv(HERE/'person-results.csv',index=False)
    persons[persons.target>=20].to_csv(HERE/'long-people.csv',index=False);runs.to_csv(HERE/'partition-metrics.csv',index=False)
    pairs=[('PlusSitAll','MatchedSpeedTurn'),('PlusSitTime','MatchedSpeedTurn'),('PlusSitMotion','MatchedSpeedTurn'),
           ('ShortSpeedTurn','MatchedSpeedTurn'),('PlusSitAll','ShortSpeedTurn'),('PlusSitAll','PlusSitTime'),('PlusSitAll','PlusSitMotion')]
    contrasts=[module.contrast(persons,runs,('Excluded','Ridge',a),('Excluded','Ridge',b),band) for a,b in pairs for band in ['all','ge20','lt20']]
    save('summary.json',{'metrics':metrics,'contrasts':contrasts,'analysis_dependency_sha256':sha(HERE.parent/'20261031-opencap-leave283/assess.py')})
    pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False)
    print(pd.DataFrame(metrics)[['view','band','people','MAE','underprediction','overprediction']].to_string(index=False),flush=True)
    print('PRIMARY',json.dumps(contrasts[1]),flush=True)
if __name__=='__main__':main()
