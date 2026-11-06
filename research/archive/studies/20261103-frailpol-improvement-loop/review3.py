import sys
from pathlib import Path
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round3 import *
def check_context_model(m,X,y,cfg):
    assert m['cfg']==cfg
    im=SimpleImputer(strategy='median',keep_empty_features=True).fit(X);a=im.transform(X);assert np.array_equal(im.statistics_,m['imputer'].statistics_,equal_nan=True)
    scores=np.nan_to_num(f_regression(a,y)[0],nan=-np.inf);k=min(cfg.get('k',32),a.shape[1]);cols=np.r_[0,np.argsort(-scores[1:],kind='stable')[:max(k-1,0)]+1]
    assert np.array_equal(cols,m['columns']);assert np.array_equal(a[:,cols],m['train_X']) and np.array_equal(y,m['train_y'])
def main():
    h=H/'round03';b=B/'round03';receipt=read(h/'run-receipt.json');assert receipt['cases']==10
    for p,v in receipt['hashes'].items():assert sha(p)==v
    protocol=read(h/'protocol.json');assert protocol['source_sha256']==sha(SRC/'cohorts.json') and protocol['code_sha256']==sha(H/'round3.py');assert protocol['round1_library_sha256']==sha(H/'round1.py');assert protocol['features_sha256']==sha(b/'features.json')
    old=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');assert protocol['factory_sha256']==sha(old/'models.py')
    for w in read(old/'weights-manifest.json'):assert sha(w['path'])==w['sha256']
    rr=[r for r in read(SRC/'cohorts.json') if r['status']=='ok' and r['tug'] is not None];ids=np.array([r['id'] for r in rr]);y=np.array([r['tug'] for r in rr]);expanded={r['id']:r['features'] for r in read(b/'features.json')};byid={r['id']:{**r,'features':expanded[r['id']]} for r in rr};lookup={int(i):j for j,i in enumerate(ids)};out={};checks=[];oof=[]
    for cond in ['speed','expanded']:
      pred={k:np.zeros(len(y)) for k in ['selected','ensemble']};seen=[]
      for fold in range(5):
        stem=f'{cond}-{fold}';rec=read(b/(stem+'.json'));ctx=np.load(rec['context']);tr=ctx['train_ids'].tolist();te=ctx['test_ids'].tolist();assert tr==rec['train_ids'] and te==rec['test_ids'];assert not set(tr)&set(te) and set(tr)|set(te)==set(ids);assert all(byid[i]['fold']==fold for i in te) and all(byid[i]['fold']!=fold for i in tr)
        cols=['speed'] if cond=='speed' else list(byid[tr[0]]['features'])
        for part,ii in [('train',tr),('test',te)]:assert np.array_equal(ctx['X'+part],np.array([[byid[i]['features'][c] for c in cols] for i in ii],float),equal_nan=True)
        assert ctx['ytrain'].tolist()==[byid[i]['tug'] for i in tr];inner=read(b/(stem+'-inner.json'));scores=[]
        replay_scores=[];max_replay_delta=0.
        for ci in range(len(protocol['candidates'])):
          yy=[];pp=[];rp=[];seeninner=[]
          for r in [r for r in inner if r['candidate']==ci]:
            it,iv=set(r['train_ids']),set(r['test_ids']);assert not it&iv and it|iv==set(tr);assert sha(r['model'])==r['sha256'];ii=[tr.index(i) for i in r['test_ids']];ti=[tr.index(i) for i in r['train_ids']];model=joblib.load(r['model']);check_context_model(model,ctx['Xtrain'][ti],ctx['ytrain'][ti],protocol['candidates'][ci]);p=predict2(model,ctx['Xtrain'][ii]);assert np.allclose(p,r['prediction'],rtol=1e-6,atol=1e-6);max_replay_delta=max(max_replay_delta,float(np.max(np.abs(p-r['prediction']))));yy.extend([byid[i]['tug'] for i in r['test_ids']]);pp.extend(r['prediction']);rp.extend(p);seeninner+=r['test_ids']
          assert len(seeninner)==len(tr) and set(seeninner)==set(tr)
          order=np.argsort([tr.index(i) for i in seeninner]);yy=np.array(yy)[order];pp=np.array(pp)[order]
          assert np.array_equal(yy,ctx['ytrain'])
          e=np.abs(pp-yy);score=max(e.mean()/.9,np.percentile(e,95)/3,e[yy>=20].mean()/3);assert np.isclose(score,rec['inner_scores'][ci]['score'],atol=1e-12,rtol=1e-12);scores.append(score)
          er=np.abs(np.array(rp,dtype=float)[order]-yy);replay_scores.append(max(er.mean()/.9,np.percentile(er,95)/3,er[yy>=20].mean()/3))
        top=np.argsort(scores,kind='stable')[:3];assert top.tolist()==rec['top3'];assert np.argsort(replay_scores,kind='stable')[:3].tolist()==top.tolist();ps=[]
        for m in rec['models']:
          assert sha(m['path'])==m['sha256'];model=joblib.load(m['path']);check_context_model(model,ctx['Xtrain'],ctx['ytrain'],protocol['candidates'][m['candidate']]);ps.append(predict2(model,ctx['Xtest']))
        assert np.allclose(ps[0],rec['selected_prediction'],atol=1e-6,rtol=1e-6);assert np.allclose(np.mean(ps,axis=0),rec['ensemble_prediction'],atol=1e-6,rtol=1e-6)
        max_replay_delta=max(max_replay_delta,float(np.max(np.abs(ps[0]-rec['selected_prediction']))),float(np.max(np.abs(np.mean(ps,axis=0)-rec['ensemble_prediction']))))
        ix=[lookup[i] for i in te];pred['selected'][ix]=rec['selected_prediction'];pred['ensemble'][ix]=rec['ensemble_prediction'];seen+=te;checks.append(dict(condition=cond,fold=fold,inner_models_replayed=len(inner),final_models_replayed=3,top3=top.tolist(),replay_selection_identical=True,max_replay_prediction_delta_seconds=max_replay_delta));print('verified',stem,flush=True)
      assert len(seen)==len(ids) and set(seen)==set(ids)
      for kind,p in pred.items():
        m=metrics(y,p);m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);out[cond+'/'+kind]=m
        for i,actual,pr in zip(ids,y,p):oof.append(dict(id=int(i),condition=cond,kind=kind,actual=float(actual),prediction=float(pr)))
    save(b/'heldout.json',oof);save(h/'comparison.json',out);save(h/'verification.json',dict(status='passed',outer_cases=10,inner_models_replayed=sum(c['inner_models_replayed'] for c in checks),final_models_replayed=30,checks=checks,review_sha256=sha(Path(__file__)),comparison_sha256=sha(h/'comparison.json')))
    save(H/'progress.json',dict(status='round03_verified',goal_met=any(m['goal_met'] for k,m in out.items() if k.startswith('expanded/')),round03=out));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

