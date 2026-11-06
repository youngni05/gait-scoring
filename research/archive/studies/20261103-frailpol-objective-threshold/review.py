from pathlib import Path
import json,hashlib,sys,itertools
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,confusion_matrix
H=Path('D:/Projects/D_P/studies/20261103-frailpol-objective-threshold');B=Path('D:/gait-research-data/D_P/frailpol-objective-threshold-20261103');OLD=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');SOURCE=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103');sys.stdout.reconfigure(encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def reg(y,p):return dict(n=len(y),mae=float(abs(p-y).mean()),bias=float((p-y).mean()),rmse=float(np.sqrt(np.mean((p-y)**2))))
def binary(y,flag,p):
    robust_fp=int(sum((y==0)&flag));prefrail_fp=int(sum((y==1)&flag));y=y==2;tp=int(sum(y&flag));fp=int(sum(~y&flag));return dict(n=len(y),positive=int(sum(y)),tp=tp,fp=fp,fp_robust=robust_fp,fp_prefrail=prefrail_fp,fn=int(sum(y&~flag)),tn=int(sum(~y&~flag)),recall=tp/int(sum(y)),fpr=fp/int(sum(~y)),ppv=tp/(tp+fp) if tp+fp else None,auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)))
def bootdiff(a,b,mask=None):
    d=b-a
    if mask is not None:d=d[mask]
    rng=np.random.default_rng(20261103);v=np.array([np.mean(d[rng.integers(len(d),size=len(d))]) for _ in range(10000)])
    return dict(difference=float(d.mean()),ci95=np.quantile(v,[.025,.975]).tolist())
def main():
    receipt=read(H/'run-receipt.json');protocol=read(H/'protocol.json');assert receipt['cases']==80 and sha(H/'protocol.json')==receipt['protocol_sha256']
    for p,h in receipt['hashes'].items():assert sha(p)==h,p
    assert sha(SOURCE/'cohorts.json')==protocol['source_cohort_sha256'];assert sha(OLD/'run-receipt.json')==protocol['source_run_sha256'];assert sha(H/'followup.py')==protocol['code_sha256']
    assert sha(OLD/'models.py')==read(OLD/'run-receipt.json')['factory_sha256']
    allrows=read(SOURCE/'cohorts.json');byid={r['id']:r for r in allrows};grid=read(OLD/'protocol.json')['LGBM']['grid'];cfgs=[dict(zip(grid,x)) for x in itertools.product(*grid.values())];verified=[];result={};oof=[]
    for target in ['frailty','tug']:
      rows=[r for r in allrows if r['status']=='ok' and r[target] is not None];ids=[r['id'] for r in rows];lookup={i:j for j,i in enumerate(ids)};y=np.array([r[target] for r in rows]);cl=target=='frailty';preds={};flags={}
      cases=[('LGBM','original'),('TabICLv2','original'),('TabPFNv2','original'),('LGBM','weighted')] if cl else [('LGBM','raw'),('TabICLv2','raw'),('TabPFNv2','raw'),('LGBM','weighted')]
      for kind,mode in cases:
       for cond in ['speed','features']:
        key=f'{kind}/{mode}/{cond}';pp=np.zeros((len(y),3) if cl else len(y));ff={k:np.zeros(len(y),bool) for k in ['argmax','0.5','0.1','0.2']} if cl else {};seen=[]
        for fold in range(5):
          stem=f'{target}-{cond}-{fold}-{kind}-{mode}';r=read(B/'predictions'/(stem+'.json'));ctx=np.load(r['context']);assert sha(r['context'])==r['context_sha256'];assert r['protocol_sha256']==sha(H/'protocol.json');tr=ctx['train_ids'].tolist();te=ctx['test_ids'].tolist();seen+=te
          assert len(set(tr))==len(tr) and len(set(te))==len(te) and not set(tr)&set(te) and set(tr)|set(te)==set(ids)
          assert all(byid[i]['fold']==fold for i in te) and all(byid[i]['fold']!=fold for i in tr);assert ctx['y_train'].tolist()==[byid[i][target] for i in tr]
          for part,ii in [('train',tr),('test',te)]:
            cols=['speed'] if cond=='speed' else list(byid[ii[0]]['features']);assert np.array_equal(ctx['X_'+part],[[byid[i]['features'][c] for c in cols] for i in ii])
          assert r['replay_max']<=1e-6
          inner=read(B/'inner'/(stem+'.json'));sel=r['selection']
          if kind=='LGBM':
            assert sha(r['model'])==r['model_sha256'];assert len(inner)==48
            losses=[]
            for ci in range(16):
              errs=[];ws=[];seeniv=[]
              for v in [v for v in inner if v['candidate']==ci]:
                it,iv=set(v['train']),set(v['evaluation']);assert not it&iv and it|iv==set(tr);seeniv+=v['evaluation'];yy=np.array([byid[i][target] for i in v['evaluation']]);ip=np.array(v['prediction']);e=-np.log(np.clip(ip[np.arange(len(yy)),yy.astype(int)],1e-15,1)) if cl else abs(ip-yy)
                w=np.ones(len(yy))
                if mode=='weighted':w=len(yy)/(3*np.bincount(yy.astype(int),minlength=3)[yy.astype(int)]) if cl else np.where(yy>=20,4,np.where(yy>=12,2,1))
                errs.extend(e);ws.extend(w)
              assert len(seeniv)==len(tr) and set(seeniv)==set(tr);losses.append(float(np.average(errs,weights=ws)))
            assert np.allclose(losses,sel['losses'],atol=1e-12,rtol=1e-12);assert int(np.argmin(losses))==sel['candidate'] and sel['params']==cfgs[sel['candidate']]
          ix=[lookup[i] for i in te];p=np.array(r['prediction']);pp[ix]=p
          if mode=='original':
            source_rec=read(SOURCE/'predictions'/f'{target}-{cond}-{fold}-{kind}.json');source_context=np.load(source_rec['context']);assert source_context['test_ids'].tolist()==te and source_context['train_ids'].tolist()==tr;assert np.array_equal(p,source_rec['prediction'])
          if cl:
            assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1,atol=1e-5)
            picked=[v for v in inner if v['candidate']==(sel['candidate'] if sel else 0)];ip=[];iy=[];seeniv=[]
            for v in picked:
              assert not set(v['train'])&set(v['evaluation']) and set(v['train'])|set(v['evaluation'])==set(tr);seeniv+=v['evaluation'];iy.extend([byid[i]['frailty'] for i in v['evaluation']]);ip.extend(np.array(v['prediction'])[:,2])
            assert len(seeniv)==len(tr) and set(seeniv)==set(tr);ip=np.array(ip);iy=np.array(iy)
            for budget in [.1,.2]:
              candidates=np.r_[0,np.nextafter(np.unique(ip),np.inf)];expected=next(float(t) for t in candidates if np.sum((ip>=t)&(iy!=2))/sum(iy!=2)<=budget);assert expected==r['thresholds'][str(budget)];ff[str(budget)][ix]=p[:,2]>=expected
            ff['argmax'][ix]=p.argmax(axis=1)==2;ff['0.5'][ix]=p[:,2]>=.5
          verified.append(dict(case=stem,replay_max=r['replay_max']))
        assert len(seen)==len(ids) and set(seen)==set(ids);preds[key]=pp
        if cl:flags[key]=ff
      if not cl:
        for kind in ['LGBM','TabICLv2','TabPFNv2']:
         for cond in ['speed','features']:
          p=np.zeros(len(y))
          for fold in range(5):
            r=read(SOURCE/'predictions'/f'tug-{cond}-{fold}-{kind}.json');cx=np.load(r['context']);p[[lookup[int(i)] for i in cx['test_ids']]]=r['prediction']
          preds[f'{kind}/log/{cond}']=p
      metrics={};pairs={}
      for key,p in preds.items():
        if cl:metrics[key]={rule:binary(y,flag,p[:,2]) for rule,flag in flags[key].items()};metrics[key]['three_class']=dict(balanced_accuracy=float(balanced_accuracy_score(y,p.argmax(axis=1))),confusion=confusion_matrix(y,p.argmax(axis=1),labels=[0,1,2]).tolist())
        else:metrics[key]={name:reg(y[m],p[m]) for name,m in {'all':np.ones(len(y),bool),'under12':y<12,'12to20':(y>=12)&(y<20),'20plus':y>=20}.items()}
        for i,actual,pred in zip(ids,y,p):oof.append(dict(target=target,case=key,id=i,actual=float(actual),prediction=pred.tolist() if cl else float(pred),decisions={k:bool(v[lookup[i]]) for k,v in flags[key].items()} if cl else None))
      for key in preds:
        if not key.endswith('/features'):continue
        baseline=key.replace('/features','/speed')
        if cl:pairs[key]={rule:dict(recall=bootdiff(flags[baseline][rule].astype(float),flags[key][rule].astype(float),y==2),fpr=bootdiff(flags[baseline][rule].astype(float),flags[key][rule].astype(float),y!=2)) for rule in ['argmax','0.1','0.2']}
        else:pairs[key]=dict(vs_speed=bootdiff(abs(preds[baseline]-y),abs(preds[key]-y)))
      if not cl:
        for key in preds:
          kind,mode,cond=key.split('/')
          if mode!='log':pairs.setdefault(key,{})['vs_same_input_log']=bootdiff(abs(preds[f'{kind}/log/{cond}']-y),abs(preds[key]-y))
      result[target]=dict(metrics=metrics,paired=pairs)
    save(B/'heldout.json',oof);save(H/'comparison.json',result);save(H/'verification.json',dict(status='passed',cases=len(verified),max_replay_difference=max(v['replay_max'] for v in verified),checks=verified,review_code_sha256=sha(Path(__file__)),comparison_sha256=sha(H/'comparison.json')))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
