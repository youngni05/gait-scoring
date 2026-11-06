import sys
from pathlib import Path
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round1 import *
def main():
    h=H/'round01';b=B/'round01';receipt=read(h/'run-receipt.json');assert receipt['cases']==10
    for p,v in receipt['hashes'].items():assert sha(p)==v
    protocol=read(h/'protocol.json');assert protocol['source_sha256']==sha(SRC/'cohorts.json') and protocol['code_sha256']==sha(H/'round1.py')
    rr=[r for r in read(SRC/'cohorts.json') if r['status']=='ok' and r['tug'] is not None];ids=np.array([r['id'] for r in rr]);y=np.array([r['tug'] for r in rr]);byid={r['id']:r for r in rr};lookup={int(i):j for j,i in enumerate(ids)};out={};checks=[];oof=[]
    for cond in ['speed','features']:
      pred={k:np.zeros(len(y)) for k in ['selected','ensemble']};seen=[]
      for fold in range(5):
        stem=f'{cond}-{fold}';rec=read(b/(stem+'.json'));ctx=np.load(rec['context']);tr=ctx['train_ids'].tolist();te=ctx['test_ids'].tolist();assert tr==rec['train_ids'] and te==rec['test_ids'];assert not set(tr)&set(te) and set(tr)|set(te)==set(ids);assert all(byid[i]['fold']==fold for i in te) and all(byid[i]['fold']!=fold for i in tr)
        cols=['speed'] if cond=='speed' else list(byid[tr[0]]['features'])
        for part,ii in [('train',tr),('test',te)]:assert np.array_equal(ctx['X'+part],[[byid[i]['features'][c] for c in cols] for i in ii])
        assert ctx['ytrain'].tolist()==[byid[i]['tug'] for i in tr];inner=read(b/(stem+'-inner.json'));scores=[]
        for ci in range(len(protocol['candidates'])):
          yy=[];pp=[];seeninner=[]
          for r in [r for r in inner if r['candidate']==ci]:
            it,iv=set(r['train_ids']),set(r['test_ids']);assert not it&iv and it|iv==set(tr);assert sha(r['model'])==r['sha256'];ii=[tr.index(i) for i in r['test_ids']];p=predict(joblib.load(r['model']),ctx['Xtrain'][ii]);assert np.allclose(p,r['prediction'],rtol=1e-9,atol=1e-9);yy.extend([byid[i]['tug'] for i in r['test_ids']]);pp.extend(p);seeninner+=r['test_ids']
          assert len(seeninner)==len(tr) and set(seeninner)==set(tr);yy=np.array(yy);pp=np.array(pp);e=np.abs(pp-yy);score=max(e.mean()/.9,np.percentile(e,95)/3,e[yy>=20].mean()/3);assert np.isclose(score,rec['inner_scores'][ci]['score'],atol=1e-12,rtol=1e-12);scores.append(score)
        top=np.argsort(scores,kind='stable')[:3];assert top.tolist()==rec['top3'];ps=[]
        for m in rec['models']:
          assert sha(m['path'])==m['sha256'];ps.append(predict(joblib.load(m['path']),ctx['Xtest']))
        assert np.allclose(ps[0],rec['selected_prediction'],atol=1e-9,rtol=1e-9);assert np.allclose(np.mean(ps,axis=0),rec['ensemble_prediction'],atol=1e-9,rtol=1e-9)
        ix=[lookup[i] for i in te];pred['selected'][ix]=ps[0];pred['ensemble'][ix]=np.mean(ps,axis=0);seen+=te;checks.append(dict(condition=cond,fold=fold,inner_models_replayed=len(inner),final_models_replayed=3,top3=top.tolist()));print('verified',stem,flush=True)
      assert len(seen)==len(ids) and set(seen)==set(ids)
      for kind,p in pred.items():
        m=metrics(y,p);m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);out[cond+'/'+kind]=m
        for i,actual,pr in zip(ids,y,p):oof.append(dict(id=int(i),condition=cond,kind=kind,actual=float(actual),prediction=float(pr)))
    save(b/'heldout.json',oof);save(h/'comparison.json',out);save(h/'verification.json',dict(status='passed',outer_cases=10,inner_models_replayed=sum(c['inner_models_replayed'] for c in checks),final_models_replayed=30,checks=checks,review_sha256=sha(Path(__file__)),comparison_sha256=sha(h/'comparison.json')))
    save(H/'progress.json',dict(status='round01_verified',goal_met=any(m['goal_met'] for k,m in out.items() if k.startswith('features/')),round01=out));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
