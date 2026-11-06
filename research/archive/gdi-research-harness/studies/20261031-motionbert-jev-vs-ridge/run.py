"""Frozen MotionBERT inputs: direct Jev prediction versus training-only Ridge."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(HERE.parent/'20261031-jev-feasibility'),str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps')]
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from jev_client import call

SOURCES={'TUG':HERE.parent/'20261030-toronto-poma-tug/features.npz',
         'TUG_folds':HERE.parent/'20261030-tug-toronto-timeseries/input.npz',
         'GDI':HERE.parent/'20261030-gdi-fusion695/data.npz'}
ALPHAS=[1.,10.,100.,1000.]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(name,value):(HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def load_data():
    t=dict(np.load(SOURCES['TUG'],allow_pickle=False));f=dict(np.load(SOURCES['TUG_folds'],allow_pickle=False))
    folds=[]
    for pid in t['person']:
        ix=np.array([int(v)==pid for v in f['ids']]);values=np.unique(f['folds'][ix]);assert len(values)==1
        folds.append(int(values[0]));np.testing.assert_allclose(f['y'][ix],t['Y'][t['person']==pid,1][0],rtol=0,atol=1e-6)
    datasets={'TUG':{'X':t['X'],'y':t['Y'][:,1],'ids':t['person'].astype(str),'fold':np.array(folds)}}
    with np.load(SOURCES['GDI'],allow_pickle=False) as z:
        people=z['person_id'];xx=z['motionbert'];yy=z['y'];ff=z['fold']
    ids=np.unique(people);X=[];y=[];fold=[]
    for pid in ids:
        ix=people==pid;assert len(np.unique(ff[ix]))==1
        X.append(xx[ix].mean(0));y.append(yy[ix].mean());fold.append(int(ff[ix][0]))
    datasets['GDI']={'X':np.array(X),'y':np.array(y),'ids':ids,'fold':np.array(fold)}
    assert len(datasets['TUG']['y'])==12 and len(datasets['GDI']['y'])==695
    for d in datasets.values():assert d['X'].shape==(len(d['y']),512) and np.isfinite(d['X']).all()
    return datasets

def grid(dataset):return np.arange(0,61,1,dtype=float) if dataset=='TUG' else np.arange(0,141,2,dtype=float)

def encode(d,fold):
    tr=d['fold']!=fold
    scaler=StandardScaler().fit(d['X'][tr])
    return np.round(scaler.transform(d['X']),3)

def ridge_fit(x,y):
    splits=list(KFold(n_splits=3,shuffle=True,random_state=20261031).split(x))
    scores=[]
    for alpha in ALPHAS:
        errors=[]
        for tr,va in splits:
            fitted=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(x[tr],y[tr])
            errors.extend(abs(fitted.predict(x[va])-y[va]))
        scores.append(float(np.mean(errors)))
    selected=ALPHAS[int(np.argmin(scores))]
    return make_pipeline(StandardScaler(),Ridge(alpha=selected)).fit(x,y),{'alpha':selected,'inner_MAE':scores}

def payload(case,d):
    z=encode(d,case['fold']);i=case['index'];examples=case['examples'];dataset=case['dataset']
    unit='seconds for the separate timed-up-and-go test' if dataset=='TUG' else 'GDI points, averaged across left/right sides and recordings per person'
    state={'task':'Predict the held-out query target from frozen MotionBERT-Lite features and labeled training examples.',
           'feature_definition':'512 latent dimensions in a comma-separated string, same ordering for every vector. Standardized using outer training people only and rounded to 3 decimals. Dimensions are not named joint angles, speeds or clinical findings. No query target is included.',
           'target_definition':unit,
           'examples':[{'example':f'e{k}','features_csv':','.join(f'{v:.3f}' for v in z[j]),'target':float(d['y'][j])} for k,j in enumerate(examples)],
           'query':{'features_csv':','.join(f'{v:.3f}' for v in z[i])},
           'example_selection':'Eight nearest outer-training people by standardized feature distance; displayed in a fixed shuffled order. Query and all other people in its fold were excluded.'}
    values=grid(dataset)
    choices={f'value_{int(v)}':f'The query target is nearest to {v:g} {unit}.' for v in values}
    choices['unknown']='The supplied feature-label examples do not support a target estimate.'
    analogy={f'e{k}':f'The query has the most similar target-relevant latent movement pattern to example e{k}.' for k in range(len(examples))}
    analogy['unknown']='No example provides a defensible target-relevant match.'
    return {'model':'jev-1.14.0','state':state,'questions':{
      'target':{'type':'choice','instructions':'Use the provided numerical feature-label relationships to estimate the query target. Select the nearest offered numerical value. Do not treat any latent coordinate as a clinical measurement. Unknown is permitted when no estimate is supported. Do not assume a diagnosis.','criteria':choices},
      'analogue':{'type':'choice','instructions':'Compare the 512-dimensional feature vectors and their supplied target examples. Select the most relevant training example for predicting the query target, or unknown. This is a numerical pattern comparison, not visual gait interpretation.','criteria':analogy}}}

def prepare():
    assert not (HERE/'responses.jsonl').exists() and not (HERE/'protocol.json').exists()
    data=load_data();cases=[];baselines=[];full={}
    protocol={'declared_before_Jev_requests':True,'created_utc':datetime.now(timezone.utc).isoformat(),
      'question':'Can Jev replace Ridge as the predictor on the SAME frozen MotionBERT output?',
      'sources':{k:{'path':str(p),'sha256':sha(p)} for k,p in SOURCES.items()},
      'data':'TUG12 existing person-mean512 embeddings; GDI695 existing record-mean512 embeddings, bilateral/record mean target per person. Protected37/38/223 sets not used.',
      'selection':'All12 TUG;10 label-blind SHA256-ordered IDs per existing GDI fold, total50 of695. Same five outer person folds as prior work.',
      'representation':'Full512 latent dimensions serialized losslessly as comma-separated text; training-only standardization, round3, identical vectors supplied to both methods. No semantic reinterpretation or feature selection.',
      'examples':'Eight nearest outer-training people, chosen without their target values, deterministically shuffled. Same8 examples and labels for matched Ridge and Jev.',
      'Ridge':'StandardScaler plus Ridge; alpha1/10/100/1000 chosen by fixed3fold INNER MAE using only the relevant training examples. Also report Ridge on ALL outer training people.',
      'Jev':'Frozen jev-1.14.0, no customer fine-tuning. In-context labeled examples only. One request has direct numerical Choice and exemplar Choice.',
      'score_output':'Primary: chosen grid point, TUG0..60 step1; GDI0..140 step2. Unknown falls back to same8-example median. Report coverage. Also report probability-weighted grid and example-target proxies as exploratory; these are not native continuous regression or calibrated clinical scores.',
      'controls':['same-example median','same-example mean','nearest1','all-training median','same-example Ridge quantized to the same grid'],
      'limits':{'max_requests':100,'max_input_tokens':3000000,'expected_requests':62,'workers':1},
      'privacy':'Only anonymous public training feature-label examples and query features sent. Query label, person IDs, filenames and images are not sent.',
      'metrics':'Same-person MAE/RMSE, paired conditional bootstrap95, per-fold error, unknown coverage. All arms reported, no held-out prompt tuning.',
      'caveats':'Repeated development cohorts; GDI50 is a pilot subset, not all695 evaluation. In-context example use differs from fitting Ridge weights. Pretraining overlap unknown.'}
    save('protocol.json',protocol)
    with threadpool_limits(limits=2):
        for name,d in data.items():
            chosen=list(range(len(d['y']))) if name=='TUG' else [i for f in range(5) for i in sorted(np.flatnonzero(d['fold']==f),key=lambda j:hashlib.sha256(('head-pilot-'+str(d['ids'][j])).encode()).hexdigest())[:10]]
            for i in chosen:
                fold=int(d['fold'][i]);tr=np.flatnonzero(d['fold']!=fold);z=encode(d,fold)
                distance=np.sum((z[tr]-z[i])**2,axis=1);nearest=tr[np.argsort(distance,kind='stable')[:8]]
                seed=int(hashlib.sha256((name+str(d['ids'][i])).encode()).hexdigest()[:8],16)
                examples=np.random.default_rng(seed).permutation(nearest).tolist()
                case={'key':f'{name}_{d["ids"][i]}','dataset':name,'index':int(i),'person':str(d['ids'][i]),'fold':fold,'examples':examples}
                assert i not in examples and np.all(d['fold'][examples]!=fold)
                matched,selection=ridge_fit(z[examples],d['y'][examples]);pred=float(matched.predict(z[i:i+1])[0])
                if (name,fold) not in full:
                    fitted,sel=ridge_fit(z[tr],d['y'][tr]);full[name,fold]=(fitted.predict(z),sel)
                values=grid(name);yy=d['y'][examples]
                base={'key':case['key'],'dataset':name,'person':case['person'],'fold':fold,'target':float(d['y'][i]),
                      'Ridge_same8':pred,'Ridge_all_training':float(full[name,fold][0][i]),
                      'Ridge_same8_grid':float(values[np.argmin(abs(values-pred))]),
                      'median_same8':float(np.median(yy)),'mean_same8':float(np.mean(yy)),
                      'nearest1':float(d['y'][nearest[0]]),'median_all_training':float(np.median(d['y'][tr])),
                      'matched_selection':selection,'full_selection':full[name,fold][1]}
                pp=payload(case,d);serialized=json.dumps(pp,sort_keys=True).encode()
                assert len(json.dumps(pp).encode())<80000
                case['payload_sha256']=hashlib.sha256(serialized).hexdigest();case['request_bytes']=len(serialized)
                cases.append(case);baselines.append(base)
    save('cases.json',cases);save('baselines.json',baselines)
    save('preparation.json',{'cases':len(cases),'max_request_bytes':max(c['request_bytes'] for c in cases),'no_API_called':True,
                           'protocol_sha256':sha(HERE/'protocol.json'),'cases_sha256':sha(HERE/'cases.json'),'baselines_sha256':sha(HERE/'baselines.json')})
    print('Prepared',len(cases),'paired cases; max request bytes',max(c['request_bytes'] for c in cases),'; no API requests.',flush=True)

def checked_load():
    receipt=json.loads((HERE/'preparation.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text())
    for key,name in [('protocol_sha256','protocol.json'),('cases_sha256','cases.json'),('baselines_sha256','baselines.json')]:assert receipt[key]==sha(HERE/name)
    for source in protocol['sources'].values():assert sha(source['path'])==source['sha256']
    return load_data(),json.loads((HERE/'cases.json').read_text())

def execute(limit=None):
    data,cases=checked_load();path=HERE/'responses.jsonl'
    old=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    archived=HERE/'format_preflight/responses.jsonl'
    prior=[json.loads(s) for s in archived.read_text().splitlines()] if archived.exists() else []
    prior_tokens=sum(r['response']['usage']['input_tokens'] for r in prior)
    prior_attempts=len(prior)+2 if prior else 0
    done={r['key'] for r in old};tokens=sum(r['response']['usage']['input_tokens'] for r in old);count=0
    for c in cases:
        if c['key'] in done:continue
        assert len(done)+prior_attempts<100 and tokens+prior_tokens+40000<=3000000
        pp=payload(c,data[c['dataset']]);assert hashlib.sha256(json.dumps(pp,sort_keys=True).encode()).hexdigest()==c['payload_sha256']
        result=call(pp);tokens+=result['usage']['input_tokens']
        with path.open('a',encoding='utf8') as f:f.write(json.dumps({'key':c['key'],'payload_sha256':c['payload_sha256'],'response':result})+'\n')
        done.add(c['key']);count+=1
        print(c['key'],'ok; tokens',result['usage']['input_tokens'],flush=True)
        if limit and count>=limit:break
    save('execution.json',{'successful_requests':len(done),'input_tokens':tokens,'estimated_usd':tokens*.042/1e6,
                         'archived_format_preflight_successes':len(prior),'archived_input_tokens':prior_tokens,
                         'failed_format_requests':2 if prior else 0,'total_known_input_tokens':tokens+prior_tokens,
                         'total_estimated_usd':(tokens+prior_tokens)*.042/1e6,'images_or_query_labels_sent':False})

def metrics(y,p):return {'MAE':float(np.mean(abs(y-p))),'RMSE':float(np.sqrt(np.mean((y-p)**2))),'bias':float(np.mean(p-y))}

def evaluate():
    data,cases=checked_load();base={r['key']:r for r in json.loads((HERE/'baselines.json').read_text())}
    response={r['key']:r for r in [json.loads(s) for s in (HERE/'responses.jsonl').read_text().splitlines()]}
    assert len(response)==len(cases)==62;predictions=[]
    for c in cases:
        r=response[c['key']];assert r['payload_sha256']==c['payload_sha256'];ans=r['response']['answers'];b=base[c['key']].copy();fallback=b['median_same8']
        chosen=ans['target']['choice'];b['Jev_direct']=fallback if chosen=='unknown' else float(chosen.removeprefix('value_'))
        b['Jev_grid_probability_proxy']=sum(p*(fallback if k=='unknown' else float(k.removeprefix('value_'))) for k,p in ans['target']['probabilities'].items())
        b['Jev_example_probability_proxy']=sum(p*(fallback if k=='unknown' else float(data[c['dataset']]['y'][c['examples'][int(k[1:])]])) for k,p in ans['analogue']['probabilities'].items())
        b['Jev_unknown']=chosen=='unknown';b['Jev_confidence']=ans['target']['confidence']
        b['Jev_unknown_probability']=ans['target']['probabilities']['unknown'];predictions.append(b)
    methods=['Ridge_same8','Ridge_all_training','Ridge_same8_grid','median_same8','mean_same8','nearest1','median_all_training','Jev_direct','Jev_grid_probability_proxy','Jev_example_probability_proxy']
    summary={}
    for name in data:
        rows=[r for r in predictions if r['dataset']==name];y=np.array([r['target'] for r in rows]);n=len(y)
        entry={'people':n,'metrics':{m:metrics(y,np.array([r[m] for r in rows])) for m in methods},'coverage':sum(not r['Jev_unknown'] for r in rows)/n,'paired':{},'per_fold':{}}
        answered=[r for r in rows if not r['Jev_unknown']]
        entry['unknown_people']=n-len(answered)
        entry['mean_unknown_probability']=float(np.mean([r['Jev_unknown_probability'] for r in rows]))
        entry['answered_only']={'people':len(answered),'metrics':{m:metrics(np.array([r['target'] for r in answered]),np.array([r[m] for r in answered])) for m in methods} if answered else {}}
        samples=np.random.default_rng(20261031).integers(0,n,size=(10000,n))
        for method in ['Jev_direct','Jev_grid_probability_proxy','Jev_example_probability_proxy']:
            for comparator in ['Ridge_same8','Ridge_all_training']:
                delta=np.array([abs(r[method]-r['target'])-abs(r[comparator]-r['target']) for r in rows])
                entry['paired'][method+'_minus_'+comparator]={'difference':float(delta.mean()),'conditional_bootstrap95':np.quantile(delta[samples].mean(1),[.025,.975]).tolist(),'people_improved':int((delta<0).sum())}
        for f in range(5):
            rr=[r for r in rows if r['fold']==f]
            entry['per_fold'][str(f)]={m:float(np.mean([abs(r[m]-r['target']) for r in rr])) for m in methods}
        summary[name]=entry
    save('predictions.json',predictions);save('results.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','execute','evaluate']);p.add_argument('--limit',type=int);a=p.parse_args()
    {'prepare':prepare,'execute':lambda:execute(a.limit),'evaluate':evaluate}[a.mode]()
