"""Read completed six-fold seeds only; preliminary prediction audit, not replay."""
import base64
import csv
import datetime
import hashlib
import json
import statistics
from collections import defaultdict
import numpy as np
from remote import HERE, REMOTE, command


def main():
    source = 'ROOT='+repr(REMOTE)+'\n'+r'''
import pathlib,json,base64,hashlib
p=pathlib.Path(ROOT); r=p/'runs/main'
seeds=json.loads((p/'protocol.json').read_text())['seeds']
ready=[s for s in seeds if all((r/('fold%d'%f)/('seed%d'%s)/'complete.json').exists() for f in range(6))]
files={}
for s in ready:
 for f in range(6):
  folder=r/('fold%d'%f)/('seed%d'%s)
  for name in ['complete.json','predictions.npz','inner/curve.json','refit/fit_complete.json']:
   path=folder/name; data=path.read_bytes()
   files[str(path.relative_to(p))]={'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()}
print(json.dumps({'seeds':ready,'completed_combinations':len(list(r.glob('fold*/seed*/complete.json'))),'files':files}))
'''
    payload=json.loads(command(source))
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    dest=HERE/'interim'/stamp; dest.mkdir(parents=True)
    hashes={}
    for name,item in payload['files'].items():
        path=(dest/name).resolve(); assert path.is_relative_to(dest.resolve())
        data=base64.b64decode(item['data']); assert hashlib.sha256(data).hexdigest()==item['sha256']
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);hashes[name]=item['sha256']
    manifest=list(csv.DictReader((HERE/'manifest.csv').open()))
    assert len(manifest)==2953 and len({r['person'] for r in manifest})==110
    models=['mean','median','mode','rf','cnn']; metrics=[]
    for seed in payload['seeds']:
        pred={m:np.full(2953,np.nan) for m in models}; seen=set()
        for f in range(6):
            folder=dest/'runs/main'/('fold%d'%f)/('seed%d'%seed)
            sel=json.loads((folder/'complete.json').read_text()); saved=np.load(folder/'predictions.npz')
            test={i for i,r in enumerate(manifest) if int(r['fold'])==f}; outer=set(range(2953))-test
            assert set(sel['test'])==test and set(sel['outer_train'])==outer
            inner=set(sel['inner_train']); valid=set(sel['validation'])
            assert inner|valid==outer and not inner&valid
            assert valid=={i for i,r in enumerate(manifest) if int(r['fold'])==(f+1)%6}
            people=[{manifest[i]['person'] for i in ids} for ids in [inner,valid,test]]
            assert not people[0]&people[1] and not people[0]&people[2] and not people[1]&people[2]
            curve=json.loads((folder/'inner/curve.json').read_text())
            epoch=min(curve,key=lambda r:r['validation_MAE'])['epoch']
            assert epoch==sel['selected_epoch']==json.loads((folder/'refit/fit_complete.json').read_text())['epochs']
            assert sel['rf_leaf']==min(sel['rf_choices'],key=lambda r:r['inner_MAE'])['leaf']
            ids=list(map(int,saved['indices']));assert len(ids)==len(set(ids)) and set(ids)==test and not seen&test;seen|=test
            ys=defaultdict(list)
            for i in outer:ys[manifest[i]['person']].append(int(manifest[i]['y']))
            mass=[statistics.mean(sum(y==k for y in v)/len(v) for v in ys.values()) for k in range(4)]
            expected={'mean':sum(k*mass[k] for k in range(4)),'median':next(k for k in range(4) if sum(mass[:k+1])>=.5),'mode':int(np.argmax(mass))}
            for m in models:
                values=saved[m];assert np.isfinite(values).all() and (values>=0).all() and (values<=3).all()
                if m in expected:assert np.max(abs(values-expected[m]))<1e-12
                pred[m][ids]=values
        assert len(seen)==2953
        for model in models:
            for group in ['all','high_2_3','score3']:
                errors=defaultdict(list)
                for i,r in enumerate(manifest):
                    y=int(r['y'])
                    if group=='high_2_3' and y<2:continue
                    if group=='score3' and y!=3:continue
                    errors[r['person']].append(float(pred[model][i])-y)
                avg=lambda fn:statistics.mean(statistics.mean(fn(e) for e in v) for v in errors.values())
                metrics.append(dict(seed=seed,model=model,group=group,people=len(errors),walks=sum(map(len,errors.values())),MAE=avg(abs),bias=avg(lambda e:e),large_error_rate=avg(lambda e:abs(e)>=2)))
    aggregate=[]
    for m in models:
        for g in ['all','high_2_3','score3']:
            rows=[r for r in metrics if r['model']==m and r['group']==g]
            if not rows:continue
            aggregate.append(dict(model=m,group=g,people=rows[0]['people'],walks=rows[0]['walks'],mae=statistics.mean(r['MAE'] for r in rows),mae_min=min(r['MAE'] for r in rows),mae_max=max(r['MAE'] for r in rows),large_error_rate=statistics.mean(r['large_error_rate'] for r in rows),bias=statistics.mean(r['bias'] for r in rows)))
    result=dict(status='preliminary_not_checkpoint_replayed',checked_at=datetime.datetime.now().astimezone().isoformat(),seeds=payload['seeds'],completed_combinations=payload['completed_combinations'],sha256=hashes,checks=['file_hash','person_disjoint_splits','2953_walk_coverage_per_seed','inner_epoch_and_RF_selection','training_only_baselines'],metrics=metrics,aggregate=aggregate)
    (dest/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2));print(str(dest/'analysis.json'))


if __name__=='__main__':main()
