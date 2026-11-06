from study import *
def avg(values,ids):return float(pd.DataFrame({'id':ids,'x':values}).dropna().groupby('id').x.mean().mean())
def run():
    rows=[];matched=[]
    for target,seed in itertools.product(['sppb','tug'],SEEDS):
        frames=[];probs=[];warns=[]
        for f in range(5):
            tag=f'{target}-f{f}-s{seed}';frames.append(pd.read_csv(STORE/'folds'/f'{tag}-keys.csv.gz'));z=np.load(STORE/'folds'/f'{tag}.npz');probs.append(z['probabilities']);warns.append(z['warnings'])
        d=pd.concat(frames,ignore_index=True);order=d.sort_values(['fold']+KEY).index.to_numpy();d=d.iloc[order].reset_index(drop=True);prob=np.concatenate(probs,axis=2)[:,:,order];warn=np.concatenate(warns,axis=2)[:,:,order];ids=d.participantid.to_numpy();y=d['class'].to_numpy();truth=np.eye(4)[y];low=y==old.lowclass(target)
        pairs=pd.read_csv(ref.STORE/f'matches-{target}.csv.gz');a=pairs.positive_index.to_numpy();b=pairs.negative_index.to_numpy();assert (y[a]==old.lowclass(target)).all() and (y[b]!=old.lowclass(target)).all()
        for i,scenario in enumerate(SCENARIOS):
            for j,name in enumerate(METHODS):
                pr=prob[i,j];pred=pr.argmax(1);warning=warn[i,j];recalls=[avg((pred[y==c]==c).astype(float),ids[y==c]) for c in range(4)]
                rows.append(dict(target=target,seed=seed,scenario=scenario,method=name,accuracy=avg(pred==y,ids),balanced_recall=float(np.mean(recalls)),brier=avg(((pr-truth)**2).sum(1),ids),extreme_error=avg((abs(pred-y)==3).astype(float),ids),low_miss=avg(np.where(low,pred!=old.lowclass(target),np.nan),ids),fpr10_miss=avg(np.where(low,~warning,np.nan),ids),fpr10_fpr=avg(np.where(~low,warning,np.nan),ids),**{f'recall_{c}':recalls[c] for c in range(4)}))
                risk=pr[:,old.lowclass(target)];delta=risk[a]-risk[b];matched.append(dict(target=target,seed=seed,scenario=scenario,method=name,pair_order_accuracy=avg((delta>0).astype(float)+.5*(delta==0),ids[a]),pairs=len(a)))
    m=pd.DataFrame(rows);mt=pd.DataFrame(matched);assert len(m)==280 and len(mt)==280;m.to_csv(HERE/'metrics-by-seed.csv',index=False);mt.to_csv(HERE/'matched-by-seed.csv',index=False)
    contrasts=[]
    for target,model in itertools.product(['sppb','tug'],['FixedRF','DS']):
        s=m[m.target.eq(target)&m.method.eq('Speed'+model)&m.scenario.isin(['bias_m10','bias_p10'])].groupby('seed').mean(numeric_only=True);c=m[m.target.eq(target)&m.method.eq('Cadence'+model)&m.scenario.isin(['bias_m10','bias_p10'])].groupby('seed').mean(numeric_only=True);delta=c-s
        contrasts.append(dict(target=target,model=model,accuracy_delta=float(delta.accuracy.mean()),accuracy_improved_seeds=int((delta.accuracy>0).sum()),miss_delta=float(delta.fpr10_miss.mean()),miss_reduced_seeds=int((delta.fpr10_miss<0).sum()),fpr_delta=float(delta.fpr10_fpr.mean()),fpr_reduced_seeds=int((delta.fpr10_fpr<0).sum())))
    save(HERE/'summary.json',dict(metric_rows=280,matched_rows=280,metrics=m.groupby(['target','scenario','method']).mean(numeric_only=True).reset_index().to_dict('records'),primary=contrasts,analysis_sha256=sha(__file__)));print('ANALYZED',280,280,flush=True)
if __name__=='__main__':run()
