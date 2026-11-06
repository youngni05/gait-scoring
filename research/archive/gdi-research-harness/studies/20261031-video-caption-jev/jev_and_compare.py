"""Text-only Jev interpretation and paired, participant-separated prediction."""
import argparse
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
from bootstrap import HERE,ROOT,BIG,sha
sys.path[:0]=[str(HERE.parent/'20261031-jev-feasibility'),str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps')]
from jev_client import call
import numpy as np

TRAITS={
 'asymmetry':'Visible left-right difference in stepping or movement.',
 'irregularity':'Visible irregular or interrupted progression of steps.',
 'restriction':'Visible restriction of lower-limb movement.',
 'trunk_motion':'Visible trunk lean or sway during movement.',
 'measurement_problem':'Missing, ambiguous or unreliable evidence limiting movement assessment.',
 'walking_aid':'Actual use of a cane, walker or another person for physical support, excluding unused objects in the room.',
 'pause':'An actual pause or halt in progression within this observed interval.',
 'turn':'An actual change of walking direction within this interval.',
 'reduced_foot_clearance':'Visible dragging or unusually little foot clearance, rather than an unseen foot.'}
CHOICES={'absent':'The supplied observations support absence or low magnitude.',
         'present':'The supplied observations explicitly support presence or high magnitude.',
         'uncertain':'The supplied observations are ambiguous or intermediate.',
         'unknown':'Relevant observations are missing or cannot establish this property.'}
VARIANTS=['numeric','focused','generic','static','combined']

def save(n,v):(HERE/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def questions():
    return {name:{'type':'choice','instructions':description+' Judge only supplied evidence. '
       'A generated caption is an unverified observation, not a clinical conclusion. Do not invent omitted actions. '
       'A normalized coordinate is not a metric speed or anatomical angle. Do not infer a test score or diagnosis.',
       'criteria':CHOICES} for name,description in TRAITS.items()}

def requests():
    d=json.loads((HERE/'numeric_features.json').read_text());captions=[json.loads(s) for s in (HERE/'captions.jsonl').read_text().splitlines()]
    assert len(captions)==36 and len(d['people'])==12
    assert len({(c['person'],c['variant']) for c in captions})==36
    for c in captions:
        assert c['clinical_labels_read'] is False
        assert c['protocol_sha256']==sha(HERE/'protocol.json')
        assert c['frames_sha256']==sha(BIG/'frames'/f"person{c['person']:02d}.npy")
    assert d['clinical_labels_read'] is False and d['protocol_sha256']==sha(HERE/'protocol.json')
    lookup={(c['person'],c['variant']):c for c in captions};out=[]
    joint=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
           'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
    names=[f'{j}_{axis}_{stat}' for stat in ['mean','std','mean_absolute_adjacent_change'] for j in joint for axis in ['normalized_x','normalized_y','confidence']]
    for i,pid in enumerate(d['people']):
        for variant in VARIANTS:
            context={'observation':'4.1-second walking-camera interval. No timed-up-and-go task or clinical labels were supplied.'}
            if variant in ['numeric','combined']:context['numeric_features']=dict(zip(names,d['X'][i]))
            if variant!='numeric':context['automatic_visual_description']=lookup[pid,'focused' if variant=='combined' else variant]['caption']
            if variant=='combined':
                context['evidence_handling']='Compare the two sources. Zero joint confidence means no detected joint, not zero real movement. The caption may invent actions. If missing coordinates prevent corroboration, acknowledge the limitation; if the sources disagree, do not silently treat the caption as verified.'
            out.append({'person':pid,'variant':variant,'payload':{'model':'jev-1.14.0','state':context,'questions':questions()}})
    return out

def run():
    path=HERE/'jev_responses.jsonl';old=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done={(r['person'],r['variant']) for r in old};tokens=sum(r['response']['usage']['input_tokens'] for r in old)
    extension=HERE/'budget_extension.json'
    token_cap=json.loads(extension.read_text())['new_cap'] if extension.exists() else 200000
    for r in requests():
        if (r['person'],r['variant']) in done:continue
        assert len(done)<60 and tokens+10000<=token_cap
        response=call(r['payload']);tokens+=response['usage']['input_tokens']
        record={'person':r['person'],'variant':r['variant'],
                'payload_sha256':hashlib.sha256(json.dumps(r['payload'],sort_keys=True).encode()).hexdigest(),'response':response}
        with path.open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n')
        done.add((r['person'],r['variant']));print(r['person'],r['variant'],'ok',flush=True)
    save('jev_execution.json',{'requests':len(done),'input_tokens':tokens,'estimated_usd_at_current_rate':tokens*.042/1e6,
                              'raw_video_sent_to_Jev':False,'labels_sent':False})

def model(family):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    return make_pipeline(StandardScaler(),Ridge(alpha=100.) if family=='Ridge' else
         RandomForestRegressor(n_estimators=128,min_samples_leaf=2,random_state=20261031,n_jobs=1))

def metrics(y,p):
    return {'MAE':float(np.abs(y-p).mean()),'RMSE':float(np.sqrt(np.mean((y-p)**2))),
            'bias':float((p-y).mean())}

def evaluate():
    from threadpoolctl import threadpool_limits
    d=json.loads((HERE/'numeric_features.json').read_text());people=np.array(d['people']);X=np.array(d['X'])
    # First clinical-label use in this pipeline occurs only after all captions are frozen.
    assert len((HERE/'captions.jsonl').read_text().splitlines())==36
    label_read_started=datetime.now(timezone.utc).isoformat()
    source=HERE.parent/'20261030-tug-toronto-timeseries/input.npz';old=dict(np.load(source,allow_pickle=False))
    print('source keys/shapes',{k:v.shape for k,v in old.items()},flush=True)
    y=[];fold=[]
    for pid in people:
        # Existing input IDs may be numeric or zero-padded strings.
        mask=np.array([int(str(x).replace('OAW',''))==int(pid) for x in old['ids']])
        assert mask.any() and len(np.unique(old['y'][mask]))==1 and len(np.unique(old['folds'][mask]))==1
        y.append(float(old['y'][mask][0]));fold.append(int(old['folds'][mask][0]))
    y=np.array(y);fold=np.array(fold);rows=[json.loads(s) for s in (HERE/'jev_responses.jsonl').read_text().splitlines()]
    lookup={(r['person'],r['variant']):r for r in rows};assert len(lookup)==60
    J={variant:np.array([[lookup[pid,variant]['response']['answers'][q]['probabilities'][c] for q in TRAITS for c in CHOICES] for pid in people])
       for variant in VARIANTS}
    views={'numeric':X}
    for v in J:
        views['Jev_'+v]=J[v]
        views['numeric_plus_Jev_'+v]=np.column_stack([X,J[v]])
    preds={'train_median':np.array([np.median(y[fold!=f]) for f in fold])}
    with threadpool_limits(limits=2):
        for family in ['Ridge','RF']:
            for view,xx in views.items():
                pp=np.full(len(y),np.nan)
                for f in sorted(set(fold)):
                    tr=fold!=f;te=fold==f;assert not set(people[tr])&set(people[te])
                    pp[te]=model(family).fit(xx[tr],y[tr]).predict(xx[te])
                assert np.isfinite(pp).all();preds[family+'_'+view]=pp
            print(family,'complete',flush=True)
        # Prespecified diagnostic negative control: shuffle within train/test independently.
        shuffled=[]
        for seed in range(50):
            rng=np.random.default_rng(20261031+seed);pp=np.full(len(y),np.nan)
            for f in sorted(set(fold)):
                tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
                a=np.column_stack([X[tr],J['focused'][rng.permutation(tr)]])
                b=np.column_stack([X[te],J['focused'][rng.permutation(te)]])
                pp[te]=model('Ridge').fit(a,y[tr]).predict(b)
            shuffled.append(pp)
    results={'metrics':{k:metrics(y,p) for k,p in preds.items()},'paired':{},
             'shuffle_Ridge_MAE':[metrics(y,p)['MAE'] for p in shuffled],
             'people':len(y),'target':'TUG_seconds','label_source_sha256':sha(source),
             'caps_sha256':sha(HERE/'captions.jsonl'),'input_sha256':sha(HERE/'numeric_features.json')}
    rng=np.random.default_rng(20261031);samples=rng.integers(0,len(y),size=(10000,len(y)))
    for family in ['Ridge','RF']:
        base=preds[family+'_numeric']
        for v in J:
            key=family+'_numeric_plus_Jev_'+v;delta=abs(preds[key]-y)-abs(base-y)
            results['paired'][key]={'difference':float(delta.mean()),'conditional_bootstrap95':np.quantile(delta[samples].mean(1),[.025,.975]).tolist(),
                                    'people_improved':int((delta<0).sum()),'people_worse':int((delta>0).sum())}
    save('results.json',results)
    save('predictions.json',{'people':people.tolist(),'target':y.tolist(),'fold':fold.tolist(),
                            'predictions':{k:p.tolist() for k,p in preds.items()},'shuffle_Ridge_predictions':[p.tolist() for p in shuffled]})
    save('label_access.json',{'caption_generation_complete_before_label_read':True,'source':str(source),'sha256':sha(source),
                             'label_read_started_utc':label_read_started,
                             'caption_file_mtime_utc':datetime.fromtimestamp((HERE/'captions.jsonl').stat().st_mtime,timezone.utc).isoformat(),
                             'caveat':'Small reused development cohort; source-model pretraining overlap unknown; no clinical validation.'})
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','evaluate']);a=p.parse_args()
    run() if a.mode=='run' else evaluate()
