from study import *
def main():
    helper=mod('old_aggregate',HERE.parent/'20261031-opencap-leave283/assess.py');protocol=json.loads((HERE/'protocol.json').read_text());summary={};cases=[]
    for c,info in protocol['cohorts'].items():
        out=HERE/c;p=pd.read_csv(out/'predictions.csv');p['scenario']=c;p['family']='Ridge';part,persons,runs,metrics=helper.aggregate(p)
        for n,f in [('partition-person',part),('person-results',persons),('partition-metrics',runs),('metrics',pd.DataFrame(metrics))]:f.to_csv(out/f'{n}.csv',index=False)
        pairs=[(f'{a}_{b}',f'{z}_{b}') for b in [10,20,30] for a,z in [('Withheld','Gait'),('Candidate','Withheld'),('Candidate','Gait')]]
        contrasts=[helper.contrast(persons,runs,(c,'Ridge',a),(c,'Ridge',z),band) for a,z in pairs for band in ['all','ge20']]
        summary[c]={'metrics':metrics,'contrasts':contrasts};save(c+'/summary.json',summary[c])
        for (view,id),g in p[p.id.isin(['sub-283','sub-288','sub-397'])].groupby(['view','id']):
            q=g.prediction.to_numpy();e=q-g.target.to_numpy();cases.append({'cohort':c,'view':view,'id':id,'target':float(g.target.iloc[0]),'mean':float(q.mean()),'sd':float(q.std()),'min':float(q.min()),'max':float(q.max()),'MAE':float(abs(e).mean())})
        print(c,pd.DataFrame(metrics).query("view in ['Gait_20','Withheld_20','Candidate_20'] and band in ['all','ge20']")[['view','band','people','MAE','underprediction','overprediction']].to_string(index=False),flush=True)
    save('summary.json',summary);save('case-predictions.json',cases)
if __name__=='__main__':main()
