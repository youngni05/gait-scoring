"""Fresh artifact fingerprints, fixed splits, original selection and metrics audit."""
from pathlib import Path
import sys
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round1 import B,SRC,read,save,sha,np,StratifiedKFold
def main():
    rows=[r for r in read(SRC/'cohorts.json') if r['status']=='ok' and r['tug'] is not None];ids=np.array([r['id'] for r in rows]);y=np.array([r['tug'] for r in rows]);fold=np.array([r['fold'] for r in rows]);assert len(ids)==594 and sum(y>=20)==9
    summary={}
    for rd in ['round01','round02','round03','round04']:
        h=H/rd;b=B/rd;receipt=read(h/'run-receipt.json');verification=read(h/'verification.json');assert verification['status']=='passed';hashcount=0
        for p,v in receipt['hashes'].items():assert sha(p)==v;hashcount+=1
        assert sha(h/'comparison.json')==verification['comparison_sha256'];assert sha(H/('review'+str(int(rd[-2:]))+'.py'))==verification['review_sha256']
        protocol=read(h/'protocol.json');assert protocol['source_sha256']==sha(SRC/'cohorts.json');configs=protocol['candidates'];conditions=protocol.get('conditions',['speed','features']);comparisons={};inner_count=0;final_count=0
        for cond in conditions:
            pp={k:np.zeros(len(y)) for k in ['selected','ensemble']};seen=[]
            for f in range(5):
                rec=read(b/(f'{cond}-{f}.json'));ctx=np.load(rec['context']);tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);assert rec['train_ids']==ctx['train_ids'].tolist()==ids[tr].tolist();assert rec['test_ids']==ctx['test_ids'].tolist()==ids[te].tolist();assert np.array_equal(ctx['ytrain'],y[tr]);assert rec['protocol_sha256']==sha(h/'protocol.json')
                inner=read(b/(f'{cond}-{f}-inner.json'));splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));scores=[]
                for ci in range(len(configs)):
                    ip=np.zeros(len(tr))
                    for k,(it,iv) in enumerate(splits):
                        rr=[q for q in inner if q['candidate']==ci and q['inner']==k];assert len(rr)==1;q=rr[0];assert q['train_ids']==ids[tr[it]].tolist() and q['test_ids']==ids[tr[iv]].tolist();assert sha(q['model'])==q['sha256'];ip[iv]=q['prediction'];inner_count+=1
                    e=np.abs(ip-y[tr]);s=max(e.mean()/.9,np.quantile(e,.95)/3,e[y[tr]>=20].mean()/3);assert np.isclose(s,rec['inner_scores'][ci]['score'],atol=1e-12,rtol=1e-12);scores.append(s)
                assert np.argsort(scores,kind='stable')[:3].tolist()==rec['top3']
                for q in rec['models']:assert sha(q['path'])==q['sha256'];final_count+=1
                for k in pp:pp[k][te]=rec[k+'_prediction']
                seen.extend(ids[te])
            assert len(seen)==len(ids) and set(seen)==set(ids)
            for k,p in pp.items():
                e=np.abs(p-y);m=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()));m['goal_met']=m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3;old=read(h/'comparison.json')[cond+'/'+k]
                for key in ['mae','p95','slow_mae','slow_bias']:assert np.isclose(m[key],old[key],atol=1e-12,rtol=1e-12)
                assert m['goal_met']==old['goal_met'];comparisons[cond+'/'+k]=m
        assert inner_count==verification['inner_models_replayed'] and final_count==verification['final_models_replayed'];summary[rd]=dict(status='passed',artifact_hashes_checked=hashcount,inner_model_fingerprints_and_splits_checked=inner_count,final_model_fingerprints_checked=final_count,prior_replay_record_current=True,metrics= comparisons);print(rd,inner_count,final_count,'goals',any(m['goal_met'] for m in comparisons.values()),flush=True)
    save(H/'resume-audit.json',dict(status='passed',scope='Fresh fingerprints, exact frozen IDs/folds/inner partitions, original selection and independent metrics; relies on unchanged completed saved-model replay records, not new foundation inference.',rounds=summary))
if __name__=='__main__':main()
