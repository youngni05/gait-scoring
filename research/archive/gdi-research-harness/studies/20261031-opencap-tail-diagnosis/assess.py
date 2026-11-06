from analyze import *

def main():
    p=pd.read_csv(HERE/'learning-predictions.csv');p['setting']=p.condition.str.replace(r'_s\d+$','',regex=True)
    e=p.prediction-p.target;p['MAE']=abs(e);p['under']=np.maximum(-e,0);p['over']=np.maximum(e,0)
    metrics=['MAE','under','over'];person=p.groupby(['setting','view','id']).agg(target=('target','first'),**{k:(k,'mean') for k in metrics}).reset_index()
    person.to_csv(HERE/'learning-person-errors.csv',index=False);summary=[]
    large={r['id'] for r in json.loads((HERE/'person-audit.json').read_text()) if r['large_under']}
    def select(g,band):return g if band=='all' else g[g.target>=20] if band=='long8' else g[g.id.isin(large)]
    for (setting,view),g in person.groupby(['setting','view']):
        for band in ['all','long8','large_under4']:
            z=select(g,band);train=p[(p.setting==setting)&(p.view==view)][['repeat','fold','condition','train_people','train_long']].drop_duplicates()
            summary.append({'setting':setting,'view':view,'band':band,'people':len(z),**{k:float(z[k].mean()) for k in metrics},'train_people_mean':float(train.train_people.mean()),'train_people_min':int(train.train_people.min()),'train_people_max':int(train.train_people.max()),'train_long_mean':float(train.train_long.mean())})
    pd.DataFrame(summary).to_csv(HERE/'learning-summary.csv',index=False)
    contrasts=[]
    for view in ['Baseline_20','Pooled_20']:
        for other in ['Fraction50','Fraction75','NoLong','DropMatchedShort']:
            for band in ['all','long8','large_under4']:
                a=select(person[(person.view==view)&(person.setting=='Full')],band).set_index('id').sort_index()
                b=select(person[(person.view==view)&(person.setting==other)],band).set_index('id').loc[a.index]
                ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
                c={'view':view,'a':'Full','b':other,'band':band}
                for k in metrics:
                    delta=(a[k]-b[k]).to_numpy();c[k]={'difference':float(delta.mean()),'conditional95':np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),'people_better':int(sum(delta<0))}
                contrasts.append(c)
    save('learning-contrasts.json',contrasts)
    print(pd.DataFrame(summary).query("band in ['all','long8']").to_string(index=False),flush=True)
if __name__=='__main__':main()
