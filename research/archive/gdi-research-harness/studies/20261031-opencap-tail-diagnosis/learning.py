from analyze import *
from threadpoolctl import threadpool_limits
ALPHAS=[.1,1.,10.,100.]
VIEWS=['Baseline_20','Pooled_20']

def fit(x,y,tr,te,alpha):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    sc=StandardScaler().fit(x[tr,:-1]);model=Ridge(alpha=alpha,solver='svd').fit(sc.transform(x[tr,:-1]),y[tr]-x[tr,-1])
    pred=np.maximum(model.predict(sc.transform(x[te,:-1])),0)+x[te,-1]
    return pred,(sc,model)

def prepare():
    assert not (HERE/'learning-protocol.json').exists()
    out=PRIOR/'exclude283';d=np.load(out/'input.npz');y=d['y'];plans=[]
    for r in json.loads((out/'splits.json').read_text()):
        for f in r['splits']:
            tr=np.array(f['train']);treatments={'Full':tr,'NoLong':tr[y[tr]<20]}
            for seed in [7310,7311,7312]:
                rng=np.random.default_rng(np.random.SeedSequence([seed,r['repeat'],f['fold']]))
                short=rng.permutation(tr[y[tr]<20]);tail=rng.permutation(tr[y[tr]>=20])
                for pct in [50,75]:
                    chosen=set(np.r_[short[:int(len(short)*pct/100)],tail[:max(1,int(len(tail)*pct/100))]].tolist())
                    treatments[f'Fraction{pct}_s{seed}']=np.array([k for k in tr if k in chosen])
                remove=set(short[:len(tail)].tolist());treatments[f'DropMatchedShort_s{seed}']=np.array([k for k in tr if k not in remove])
            for name,chosen in treatments.items():
                subset=set(chosen.tolist());inner=[{k:[i for i in v[k] if i in subset] for k in ['train','validation']} for v in f['inner']]
                assert not subset&set(f['test']) and all(v['train'] and v['validation'] for v in inner)
                assert sorted(i for v in inner for i in v['validation'])==sorted(subset)
                plans.append({'repeat':r['repeat'],'fold':f['fold'],'condition':name,'train':chosen.tolist(),'test':f['test'],'inner':inner,'train_people':len(chosen),'train_long':int(sum(y[chosen]>=20))})
    save('learning-plans.json',plans)
    save('learning-protocol.json',{'primary':'diagnostic learning curves using same176 people,8long,original5person partitions,max20s windows;42.59 absent throughout',
      'conditions':'Full; stratified nested50/75percent train samples using seeds7310/7311/7312; NoLong removes trainingTUG>=20; DropMatchedShort removes equal number ofshort train people using same3seeds',
      'inner':'original inner memberships restricted to retained train people; original test people unchanged; train-only scaling,alpha0.1/1/10/100 byinnerMAE',
      'hypothesis':'whether input-conditioned long cases or general sample count help; not estimating effect of collecting unseen new people; no fitting curve to extrapolate beyond observed people',
      'metrics':'average per-person errors over5 outer repeats and3 subsamples where present; subsamples/repeats not independent people; report same8long and large-under4 selected retrospectively',
      'views':VIEWS,'source_hashes':{str((out/n).relative_to(HERE.parent)):sha(out/n) for n in ['input.npz','splits.json','predictions.csv']},'plans_sha256':sha(HERE/'learning-plans.json'),'code_sha256':sha(HERE/'learning.py')})

def main():
    prepare();(HERE/'models').mkdir(exist_ok=True)
    data=np.load(PRIOR/'exclude283/input.npz');y=data['y'];ids=data['ids'];rows=[];selected=[];inner_rows=[]
    with threadpool_limits(limits=1):
        for i,f in enumerate(json.loads((HERE/'learning-plans.json').read_text())):
            tr=np.array(f['train']);te=np.array(f['test'])
            for view in VIEWS:
                x=data[view];scores=[]
                for alpha in ALPHAS:
                    pred=np.full(len(y),np.nan)
                    for v in f['inner']:
                        it=np.array(v['train']);iv=np.array(v['validation']);pred[iv]=fit(x,y,it,iv,alpha)[0]
                    score=float(abs(pred[tr]-y[tr]).mean());scores.append(score)
                    inner_rows.append({k:f[k] for k in ['repeat','fold','condition']}|{'view':view,'alpha':alpha,'MAE':score,'predictions':pred[tr].tolist()})
                alpha=ALPHAS[int(np.argmin(scores))];pred,pair=fit(x,y,tr,te,alpha);file=f'models/r{f["repeat"]}-f{f["fold"]}-{f["condition"]}-{view}.joblib';joblib.dump(pair,HERE/file,compress=3)
                selected.append({k:f[k] for k in ['repeat','fold','condition','train_people','train_long']}|{'view':view,'alpha':alpha,'inner_MAE':min(scores),'file':file,'sha256':sha(HERE/file)})
                rows.extend({k:f[k] for k in ['repeat','fold','condition','train_people','train_long']}|{'view':view,'id':str(ids[k]),'sample':int(k),'target':float(y[k]),'prediction':float(p)} for k,p in zip(te,pred))
            if (i+1)%55==0:print('LEARNING',i+1,'/',275,flush=True)
    pd.DataFrame(rows).to_csv(HERE/'learning-predictions.csv',index=False);save('learning-inner.json',inner_rows);save('learning-selections.json',selected)
    print('TRAINED',len(selected),flush=True)
if __name__=='__main__':main()
