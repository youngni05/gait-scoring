"""Frozen, person-separated Jev feasibility pilot. No raw video leaves this process."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
from jev_client import call

TUG = HERE.parent / '20261031-opencap-walk-pattern/TUG_common6'
GDI = HERE.parent / '20261030-gdi-fusion695'
MAX_REQUESTS = 2200
MAX_TOKENS = 5_000_000
TRAITS = {
    'asymmetry': 'How large is the evidence for left-right movement asymmetry?',
    'irregularity': 'How large is the evidence for irregular timing or inconsistent movement?',
    'restriction': 'How large is the evidence for restricted lower-limb motion?',
    'trunk_motion': 'How large is the evidence for trunk or pelvis sway?',
    'measurement_problem': 'How large is the evidence that missing or unreliable measurements limit interpretation?',
}
TUG_NAMES = ['speed_mean_m_per_s', 'speed_sd_m_per_s', 'speed_adjacent_change_m_per_s',
             'cycle_seconds', 'cycle_cv', 'cycle_time_asymmetry', 'knee_cycle_rom_degrees',
             'knee_cycle_rom_asymmetry', 'hip_cycle_rom_degrees', 'hip_cycle_rom_asymmetry',
             'ankle_cycle_rom_degrees', 'ankle_cycle_rom_asymmetry', 'hip_knee_cycle_correlation',
             'pelvis_list_rom_degrees', 'lumbar_extension_rom_degrees',
             'lumbar_bending_rom_degrees', 'lumbar_rotation_rom_degrees']


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return round(float(value), 5) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def prepare():
    assert not (HERE / 'prepared.json').exists(), 'Preparation is immutable; use a new study for changed conditions.'
    sources = [TUG/'input.npz', TUG/'splits.json', TUG/'predictions.csv', GDI/'data.npz']
    hashes = {str(p): sha(p) for p in sources}
    td = dict(np.load(TUG/'input.npz', allow_pickle=False))
    plan = json.loads((TUG/'splits.json').read_text())[0]
    tfold = np.full(len(td['y']), -1)
    for split in plan['splits']:
        tr, te = split['train'], split['test']
        assert not set(tr) & set(te) and set(tr) | set(te) == set(range(len(tfold)))
        tfold[te] = split['fold']
    assert np.all(tfold >= 0)
    datasets = {}
    tseq = []
    for x in td['x']:
        indices = np.linspace(0, 180, 19, dtype=int)
        tseq.append({'columns': ['seconds', 'speed_m_per_s', 'right_knee_degrees', 'left_knee_degrees',
                                 'right_hip_degrees', 'left_hip_degrees', 'lumbar_bending_degrees'],
                     'rows': np.column_stack([indices/30, x[17,indices], x[5,indices], x[11,indices],
                                               x[2,indices], x[8,indices], x[15,indices]])})
    datasets['TUG'] = dict(ids=td['ids'], y=td['y'], fold=tfold, X=td['pattern'], names=TUG_NAMES,
                           sequences=tseq, context='Six seconds of straight walking, OpenCap multi-camera 3D-derived angles. '
                           'Cycle proxies come from knee peaks, not verified foot contacts. Missing features are not zero. '
                           'The separate timed-up-and-go task was not observed. No chair or turn observations are supplied.')
    gd = dict(np.load(GDI/'data.npz', allow_pickle=False))
    raw = gd['x'].astype(float)
    ok = raw[...,2] > 0
    feat, names = [], []
    for side, h, k, a in [('right',1,2,3), ('left',4,5,6)]:
        u, v = raw[:,:,h,:2]-raw[:,:,k,:2], raw[:,:,a,:2]-raw[:,:,k,:2]
        denom = np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
        valid = ok[:,:,h]&ok[:,:,k]&ok[:,:,a]&(denom>1e-9)
        angle = np.degrees(np.arccos(np.clip((u*v).sum(-1)/np.maximum(denom,1e-9),-1,1)))
        for stat in ['median', 'p90_minus_p10', 'valid_fraction']:
            values=[]
            for aa, vv in zip(angle, valid):
                values.append(vv.mean() if stat=='valid_fraction' else
                              (np.median(aa[vv]) if stat=='median' else np.diff(np.percentile(aa[vv],[10,90]))[0]) if vv.any() else np.nan)
            feat.append(values); names.append(side+'_projected_knee_'+stat)
    for joint, label in [(0,'pelvis'),(3,'right_ankle'),(6,'left_ankle'),(8,'thorax')]:
        for axis in [0,1]:
            for stat in ['sd','adjacent_change']:
                vals=[]
                for x,m in zip(raw[:,:,joint,axis],ok[:,:,joint]):
                    pair=m[1:]&m[:-1]
                    vals.append(np.std(x[m]) if stat=='sd' and m.any() else
                                np.mean(np.abs(np.diff(x))[pair]) if stat=='adjacent_change' and pair.any() else np.nan)
                feat.append(vals); names.append(f'{label}_normalized_axis{axis}_{stat}')
    feat.extend([raw[...,2].mean((1,2)),ok.mean((1,2))]); names.extend(['mean_pose_confidence','observed_fraction'])
    kinematic=np.asarray(feat).T
    people=sorted(set(gd['person_id'].tolist()))
    X=[]; yy=[]; folds=[]; seqs=[]; existing=[]
    for person in people:
        ix=np.flatnonzero(gd['person_id']==person)
        assert len(set(gd['fold'][ix].tolist()))==1
        X.append(np.nanmean(kinematic[ix],axis=0)); yy.append(gd['y'][ix].mean());folds.append(int(gd['fold'][ix[0]]))
        existing.append(float(gd['pls'][ix].mean()))
        first=ix[np.argsort(gd['video_id'][ix])[0]]
        indices=np.linspace(0,123,16,dtype=int)
        seqs.append({'representative_record_only': True, 'frame_indices':indices,
                     'joints':['right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle'],
                     'channels':['normalized_x','normalized_y','confidence'],
                     'values':raw[first,indices][:,[1,2,3,4,5,6],:]})
    datasets['GDI']=dict(ids=people,y=yy,fold=folds,X=X,names=names,sequences=seqs,existing_pls=existing,
                         context='124-frame 2D pose windows in normalized crop coordinates. Summary features average '
                         'available records per person. Pixel-projected knee angles are not anatomical 3D knee flexion. '
                         'No metric speed, camera calibration, depth or verified foot contacts are available. '
                         'Target is the mean of both GDI sides and available records per person, not each side separately.')
    protocol = {'declared_before_real_requests':True,'datasets':{'TUG':195,'GDI':695},
                'primary_comparison':'Fixed RF on numerical features versus same RF on numerical features plus Jev trait probabilities, numeric and narrative arms both reported.',
                'arms':['numeric','narrative','sequence subset','training-only nearest examples subset'],
                'controls':['repeat identical request','reverse choice order','old jev-1.13.0','all observations missing'],
                'main_model':'jev-1.14.0','no_model_training_at_provider':True,
                'folds':'TUG first of existing five split repeats; GDI existing five person folds; each person evaluated once',
                'GDI_target':'mean bilateral and record GDI per person, different metric from original side-record absolute errors',
                'local_models':'Median; Ridge alpha10; RF128 trees leaf5 seed20261031; identical settings for paired inputs; no held-out tuning',
                'direct_score':'Predeclared score bins with unknown option. Bin midpoint is only a diagnostic proxy, not a calibrated continuous score. Unknown falls back to training median for all-person error.',
                'subsets':'label-blind hashed ID order: first25 people per dataset; nearest6 training examples by training-scaled numerical features',
                'metrics':['MAE','RMSE','bias','paired conditional person bootstrap95','fold error differences','tail errors','coverage','response consistency'],
                'limits':{'requests':MAX_REQUESTS,'input_tokens':MAX_TOKENS,'workers':4,'retry':0},
                'success':'Exploratory evidence only: report all arms and uncertainty, no winner adoption or clinical success claim. Positive signal needs lower paired error with stability and no unexplained tail harm.',
                'privacy':'No names, IDs, videos, filenames, diagnoses, target labels in query state. Examples only from other folds with anonymous labels.',
                'source_hashes':hashes,'protected_GDI_sets_used':False,'new_GPU_training':False,
                'limitations':['Reused development cohorts','One existing split repeat','Narrative generated by deterministic templates, not a new trained captioning model',
                               'Same underlying observations; Jev adds a prior/representation, not a new measurement','No image arm or clinical validation']}
    save('protocol.json',protocol)
    save('prepared.json',clean(datasets))
    save('preparation.json',{'source_hashes':hashes,'protocol_sha256':sha(HERE/'protocol.json'),
                             'prepared_sha256':sha(HERE/'prepared.json'),'code_sha256':sha(__file__),
                             'people':{k:len(v['ids']) for k,v in datasets.items()}})
    print('Prepared TUG195 + GDI695; no real API requests.',flush=True)


def load():
    receipt=json.loads((HERE/'preparation.json').read_text())
    assert receipt['prepared_sha256']==sha(HERE/'prepared.json')
    assert receipt['protocol_sha256']==sha(HERE/'protocol.json')
    return json.loads((HERE/'prepared.json').read_text())


def bins(dataset):
    edges=[0,6,8,10,12,15,20,30,60] if dataset=='TUG' else [-1000,40,50,60,70,80,90,100,110,1000]
    mids=[3,7,9,11,13.5,17.5,25,45] if dataset=='TUG' else [35,45,55,65,75,85,95,105,115]
    criteria={f'b{i}': f'{edges[i]} <= value < {edges[i+1]}' for i in range(len(mids))}
    if dataset=='TUG': criteria['b7']='30 seconds or longer'
    else: criteria['b0']='GDI below40';criteria['b8']='GDI110 or greater'
    criteria['unknown']='The supplied evidence does not support assigning a score interval.'
    return edges,mids,criteria


def questions(dataset, reverse=False):
    q={name:{'type':'choice','instructions':instruction+' Judge only the supplied observations and their stated units and limitations. '
              'Do not infer absent motion. Low measurement_problem means adequate evidence, not clinical validity.',
              'criteria':{'low':'Low evidence or magnitude.','middle':'Intermediate evidence or magnitude.',
                          'high':'High evidence or magnitude.','unknown':'Not enough relevant observations to judge.'}}
       for name,instruction in TRAITS.items()}
    _,_,criteria=bins(dataset)
    q['target_band']={'type':'choice','instructions':('Predict separate TUG completion time in seconds from the observed walking.' if dataset=='TUG' else
                        'Predict person-mean Gait Deviation Index. Reference normal mean100; lower means greater deviation. '
                        'This is not a0-to100 score and not a diagnosis.')+
                        ' This is an exploratory prediction. Choose unknown if evidence does not support an interval. '
                        'Any provided examples are separate training people. Never claim a directly measured target.', 'criteria':criteria}
    if reverse:
        for value in q.values(): value['criteria']=dict(reversed(list(value['criteria'].items())))
    return q


def feature_dict(d, x):
    return {name: v for name,v in zip(d['names'],clean(np.asarray(x,dtype=float)))}


def payload(data, dataset, index, arm='numeric', reverse=False, model='jev-1.14.0'):
    d=data[dataset];X=np.asarray(d['X'],float);fold=np.asarray(d['fold']);tr=np.flatnonzero(fold!=fold[index])
    # Reference statistics and example selection use training people only.
    quantiles=np.nanpercentile(X[tr],[25,50,75],axis=0)
    values=feature_dict(d,X[index])
    # Main arms are strictly sample-local: no cohort reference that could make a
    # training person's representation depend on an outer-test person's inputs.
    state={'context':d['context']}
    if arm=='missing':
        state={'context':d['context'],'observations':'All observations missing. No gait, speed, joint, timing or quality measurements are available.'}
    elif arm=='narrative':
        state['observations']=['The measured '+name.replace('_',' ')+' is '+(str(v)+'.' if v is not None else 'unavailable.') for name,v in values.items()]
    elif arm=='sequence':
        state['measurements']=values; state['sampled_time_series']=d['sequences'][index]
        state['sampling_note']='Subsampled observed sequence; omitted frames are not evidence of stillness.'
    elif arm=='examples':
        state['measurements']=values
        med=quantiles[1];scale=np.maximum(quantiles[2]-quantiles[0],1e-6)
        filled=np.where(np.isfinite(X),X,med)
        dist=np.mean(((filled[tr]-filled[index])/scale)**2,axis=1)
        near=tr[np.argsort(dist,kind='stable')[:6]]
        assert index not in near and all(fold[j]!=fold[index] for j in near)
        state['training_examples']=[{'measurements':feature_dict(d,X[j]),'target':d['y'][j]} for j in near]
    else:state['measurements']=values
    return {'model':model,'state':state,'questions':questions(dataset,reverse)}


def jobs(data):
    out=[]
    for name,d in data.items():
        for i in range(len(d['ids'])):
            for arm in ['numeric','narrative']:out.append((name,i,arm,False,'jev-1.14.0',arm))
        subset=sorted(range(len(d['ids'])),key=lambda i:hashlib.sha256(str(d['ids'][i]).encode()).hexdigest())[:25]
        for i in subset:
            for arm,reverse,model,tag in [('sequence',False,'jev-1.14.0','sequence'),('examples',False,'jev-1.14.0','examples'),
                ('numeric',False,'jev-1.14.0','repeat'),('numeric',True,'jev-1.14.0','reverse'),
                ('numeric',False,'jev-1.13.0','old_model')]:out.append((name,i,arm,reverse,model,tag))
        for i in subset[:5]:out.append((name,i,'missing',False,'jev-1.14.0','missing'))
    assert len(out)<=MAX_REQUESTS
    return out


def job_id(job):return f'{job[0]}:{job[1]}:{job[5]}'


def run(limit=None):
    data=load(); todo=jobs(data);path=HERE/'responses.jsonl'
    previous=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done={r['job'] for r in previous};todo=[j for j in todo if job_id(j) not in done]
    if limit:todo=todo[:limit]
    used=sum(r.get('response',{}).get('usage',{}).get('input_tokens',0) for r in previous)
    attempted=len(previous);consecutive=0;start=time.monotonic()
    def execute(job):
        name,i,arm,reverse,model,tag=job;p=payload(data,name,i,arm,reverse,model)
        record={'job':job_id(job),'dataset':name,'index':i,'arm':tag,
                'payload_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,allow_nan=False).encode()).hexdigest()}
        try:record['response']=call(p);record['status']='ok'
        except RuntimeError as err:record.update(status='error',reason=str(err))
        return record
    with path.open('a',encoding='utf8') as out, ThreadPoolExecutor(max_workers=4) as executor:
        for start_index in range(0,len(todo),4):
            # Reserve conservative per-request headroom before each four-request batch.
            if attempted+4>MAX_REQUESTS or used+100000>MAX_TOKENS:break
            batch=todo[start_index:start_index+4]
            for result in executor.map(execute,batch):
                out.write(json.dumps(result,allow_nan=False)+'\n');out.flush();attempted+=1
                used+=result.get('response',{}).get('usage',{}).get('input_tokens',0)
                consecutive=consecutive+1 if result['status']=='error' else 0
            if start_index%80==0:print(json.dumps({'attempted':attempted,'input_tokens':used,'last_status':result['status'],
                                                 'elapsed_seconds':round(time.monotonic()-start)}),flush=True)
            if consecutive>=3:break
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    save('execution.json',{'attempted':len(rows),'ok':sum(r['status']=='ok' for r in rows),
                           'errors':sum(r['status']!='ok' for r in rows),'planned':len(jobs(data)),
                           'input_tokens':used,'input_cost_usd_at_0_042_per_million':used*.042/1e6,
                           'all_planned_attempted':len({r['job'] for r in rows})==len(jobs(data))})
    print(json.dumps(json.loads((HERE/'execution.json').read_text())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:run(a.limit)
