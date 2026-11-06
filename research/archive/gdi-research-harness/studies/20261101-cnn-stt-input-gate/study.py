"""Frozen-base, path-isolated input gates; outer labels are evaluation-only."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np

H = Path(__file__).resolve().parent
ROOT = H.parents[2]
S = H.parent / '20261030-gpu-motionbert-study'
sys.path.insert(0, str(ROOT / 'outputs/gdi-ridge-experiment-20261016/deps'))
from scipy.optimize import minimize
from scipy.special import expit, logit

OLD = Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/robust-aggregation692/cache')
OUTER = OLD.parent.parent / 'probability-bands692/cache/outer'
O = H / 'analysis'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}

def weights(ids):
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w / w.sum()

def person_values(v, ids):
    return np.array([v[ids == q].mean() for q in np.unique(ids)])

def mae(y, p, ids):
    return float(weights(ids) @ np.abs(y-p).mean(1))

def boundary(*groups):
    for i, a in enumerate(groups):
        for b in groups[i+1:]:
            if set(a) & set(b): raise ValueError('Person leakage')

def source_identity(proof, sources):
    matches=[p for p in proof['source_files'] if Path(p).name=='identity.json']
    assert len(matches)==1
    p=matches[0];digest=sha(p);assert digest==proof['source_files'][p]
    sources[p]=digest
    return read(p)

FEATURES = ['mean_level', 'between_model_mean_gap', 'cnn_abs_side_gap',
            'stt_abs_side_gap', 'between_model_side_gap', 'cnn_log_spread',
            'stt_log_spread', 'cnn_log_iqr', 'stt_log_iqr',
            'cnn_spread_side_difference', 'stt_spread_side_difference', 'log_windows']

def summarize(raw_c, raw_s, owner, rows, coef_c, coef_s):
    means, spreads, iqrs = [], [], []
    for raw, coef in [(raw_c, coef_c), (raw_s, coef_s)]:
        m, s, iq = [], [], []
        for row in rows:
            a = raw[owner == row].astype(float) * coef[0] + coef[1]
            if not len(a) or not np.isfinite(a).all(): raise ValueError('Invalid windows')
            m.append(a.mean(0)); s.append(a.std(0))
            iq.append(np.quantile(a, .75, axis=0)-np.quantile(a, .25, axis=0))
        means.append(np.array(m)); spreads.append(np.array(s)); iqrs.append(np.array(iq))
    c, s = means; dc = c[:,0]-c[:,1]; ds = s[:,0]-s[:,1]
    sc, ss = np.log1p(spreads); ic, iss = np.log1p(iqrs)
    x = np.column_stack([(c.mean(1)+s.mean(1))/2, c.mean(1)-s.mean(1),
        np.abs(dc), np.abs(ds), np.abs(dc-ds), sc.mean(1), ss.mean(1),
        ic.mean(1), iss.mean(1), np.abs(sc[:,0]-sc[:,1]),
        np.abs(ss[:,0]-ss[:,1]), np.log1p([(owner==r).sum() for r in rows])])
    return dict(x=x, cnn=c, stt=s)

def inner_data(f, j, sources):
    parts = []
    legacy_proof = read(S/'robust-aggregation692/analysis/verification.json')
    for family in ['CNN', 'STT']:
        p = OLD/family/f'fold{f}_inner{j}.npz'; proof = read(p.with_suffix('.json'))
        digest = sha(p)
        assert digest == proof['cache_sha256'] == legacy_proof['source_cache_hashes'][str(p)]
        assert sha(proof['checkpoint']) == proof['checkpoint_sha256']
        identity = source_identity(proof,sources)
        v = arrays(p); valid = v['validation_mask']; train = v['train_mask']
        rows = v['row_index'][valid]; keep = np.isin(v['window_record'], rows)
        base_people = np.unique(v['person_id'][train]); meta_people = np.unique(v['person_id'][valid])
        assert identity['train_people'] == base_people.tolist()
        assert identity['validation_people'] == meta_people.tolist()
        boundary(base_people, meta_people, identity['outer_evaluation_people'])
        assert (v['fold'][valid] == j).all() and (v['fold'] != f).all()
        sources[str(p)] = digest; sources[str(p.with_suffix('.json'))] = sha(p.with_suffix('.json'))
        sources[proof['checkpoint']] = proof['checkpoint_sha256']
        parts.append((v, rows, keep, identity))
    a, rows, ka, identity = parts[0]; b, rb, kb, ib = parts[1]
    for key in ['row_index','person_id','video_id','fold','y','train_mask','validation_mask']:
        np.testing.assert_array_equal(a[key], b[key])
    assert identity['train_people'] == ib['train_people']
    for key in ['window_record','window_start']: np.testing.assert_array_equal(a[key][ka], b[key][kb])
    d = summarize(a['window_raw'][ka],b['window_raw'][kb],a['window_record'][ka],rows,a['calibration'][0],b['calibration'][0])
    d.update(y=a['y'][a['validation_mask']], ids=a['person_id'][a['validation_mask']],
             video=a['video_id'][a['validation_mask']], rows=rows, base_people=np.array(identity['train_people']),
             outer_people=np.array(identity['outer_evaluation_people']), coef_c=a['calibration'][0],coef_s=b['calibration'][0])
    return d

def subset(d, mask):
    return {k: d[k][mask] for k in ['x','cnn','stt','y','ids']}

def fixed_weight(d):
    grid = np.linspace(0,1,21)
    score = [mae(d['y'],w*d['cnn']+(1-w)*d['stt'],d['ids']) for w in grid]
    i = min(range(21),key=lambda i:(score[i],abs(grid[i]-.5),grid[i]))
    return float(grid[i])

def blend(d, gates, family):
    if family == 'shared':
        return gates[:,0,None]*d['cnn']+(1-gates[:,0,None])*d['stt']
    m = gates[:,0]*d['cnn'].mean(1)+(1-gates[:,0])*d['stt'].mean(1)
    dc=d['cnn'][:,0]-d['cnn'][:,1]; ds=d['stt'][:,0]-d['stt'][:,1]
    diff=gates[:,1]*dc+(1-gates[:,1])*ds
    return np.column_stack([m+diff/2,m-diff/2])

def fit(d, family, lam, cfg):
    w=weights(d['ids']); mean=w@d['x']; sd=np.sqrt(w@((d['x']-mean)**2)); sd[sd<1e-6]=1.
    x=np.column_stack([np.ones(len(w)),np.clip((d['x']-mean)/sd,-cfg['feature_clip'],cfg['feature_clip'])])
    base=fixed_weight(d); offset=float(logit(np.clip(base,1e-4,1-1e-4)))
    nhead=1 if family=='shared' else 2; dim=x.shape[1]; eps=cfg['smooth_absolute_epsilon']
    def objective(theta):
        beta=theta.reshape(nhead,dim); gate=expit(x@beta.T+offset)
        pred=blend(d,gate,family); err=pred-d['y']; rho=np.sqrt(err**2+eps**2)
        v=float(w@rho.mean(1)+lam*np.sum(beta**2))
        e=err/rho*w[:,None]/2
        if family=='shared':
            dg=np.sum(e*(d['cnn']-d['stt']),axis=1)[:,None]
        else:
            dm=d['cnn'].mean(1)-d['stt'].mean(1)
            dd=(d['cnn'][:,0]-d['cnn'][:,1])-(d['stt'][:,0]-d['stt'][:,1])
            dg=np.column_stack([e.sum(1)*dm,(e[:,0]-e[:,1])*dd/2])
        grad=(dg*gate*(1-gate)).T@x+2*lam*beta
        return v,grad.ravel()
    opt=minimize(objective,np.zeros(nhead*dim),jac=True,method='L-BFGS-B',
                 options={'maxiter':500,'ftol':1e-12,'gtol':1e-7,'maxls':40})
    if not opt.success: raise RuntimeError(str(opt.message))
    return dict(family=family,lambda_=lam,mean=mean.tolist(),sd=sd.tolist(),offset=offset,
                beta=opt.x.reshape(nhead,dim).tolist(),iterations=int(opt.nit),objective=float(opt.fun)),objective

def predict(model,d,cfg):
    z=np.clip((d['x']-np.array(model['mean']))/np.array(model['sd']),-cfg['feature_clip'],cfg['feature_clip'])
    x=np.column_stack([np.ones(len(z)),z]); gates=expit(x@np.array(model['beta']).T+model['offset'])
    return blend(d,gates,model['family']),gates

def oracle(d):
    # Record-wise minimum uses true outcomes: diagnostic ceiling, never an input or deployed rule.
    hard=np.minimum(np.abs(d['cnn']-d['y']).mean(1),np.abs(d['stt']-d['y']).mean(1))
    den=d['cnn']-d['stt']; ratio=np.divide(d['y']-d['stt'],den,out=np.zeros_like(den),where=abs(den)>1e-12)
    options=np.column_stack([np.zeros(len(den)),np.ones(len(den)),np.clip(ratio,0,1)])
    losses=np.stack([np.abs(w[:,None]*d['cnn']+(1-w[:,None])*d['stt']-d['y']).mean(1) for w in options.T],axis=1)
    return dict(hard=float(weights(d['ids'])@hard),convex=float(weights(d['ids'])@losses.min(1)))

def train():
    start=time.perf_counter(); cfg=read(H/'protocol.json'); sources={}; paths=[]; audits=[]; screens=[]
    for f in range(5):
        for j in range(5):
            if f==j: continue
            d=inner_data(f,j,sources); people=np.unique(d['ids'])
            order=np.random.default_rng(cfg['seed']+100*f+j).permutation(people)
            n=int(.6*len(order)); ns=int(.2*len(order)); groups=[order[:n],order[n:n+ns],order[n+ns:]]
            boundary(d['base_people'],*groups,d['outer_people'])
            masks=[np.isin(d['ids'],g) for g in groups]; tr,sel,aud=[subset(d,m) for m in masks]
            refit=subset(d,masks[0]|masks[1]); w=fixed_weight(tr); wf=fixed_weight(refit)
            candidates=[dict(family='constant',lambda_=None,selection_mae=mae(sel['y'],w*sel['cnn']+(1-w)*sel['stt'],sel['ids']))]
            best={}; audit_models={}
            for family in cfg['families']:
                rows=[]
                for lam in cfg['l2_grid']:
                    m,_=fit(tr,family,lam,cfg); p,_=predict(m,sel,cfg)
                    row=dict(family=family,lambda_=lam,selection_mae=mae(sel['y'],p,sel['ids']))
                    rows.append(row);candidates.append(row)
                choice=min(rows,key=lambda r:(r['selection_mae'],-r['lambda_']))
                best[family]=choice
                audit_models[family]=fit(refit,family,choice['lambda_'],cfg)[0]
            chosen=min(candidates,key=lambda r:(r['selection_mae'],r['family']!='constant',-(r['lambda_'] or 0)))
            pred={'constant':wf*aud['cnn']+(1-wf)*aud['stt']}
            for family,model in audit_models.items(): pred[family]=predict(model,aud,cfg)[0]
            pred['selected']=pred[chosen['family']]
            audits.append(dict(outer=f,inner=j,people=aud['ids'].tolist(),unique_people=groups[2].tolist(),
                               errors={k:person_values(np.abs(v-aud['y']).mean(1),aud['ids']).tolist() for k,v in pred.items()}))
            fullw=fixed_weight(d); oo=oracle(d); baseline=mae(d['y'],fullw*d['cnn']+(1-fullw)*d['stt'],d['ids'])
            screens.append(dict(outer=f,inner=j,people=len(people),fixed_mae=baseline,oracle=oo,
                                oracle_hard_headroom=baseline-oo['hard'],oracle_convex_headroom=baseline-oo['convex']))
            # Selections frozen before using audit labels for this final refit; audit never selects a candidate.
            fullmodels={family:fit(d,family,best[family]['lambda_'],cfg)[0] for family in cfg['families']}
            paths.append(dict(outer=f,inner=j,base_people=d['base_people'].tolist(),meta_people=people.tolist(),
                outer_people=d['outer_people'].tolist(),split_people=[g.tolist() for g in groups],
                candidates=candidates,selected=chosen,models=fullmodels,fixed_weight=fullw,
                coef_c=d['coef_c'].tolist(),coef_s=d['coef_s'].tolist()))
            print(json.dumps(dict(trained_outer=f,inner=j,selected=chosen['family'])),flush=True)
    headroom=float(np.mean([a['oracle_convex_headroom'] for a in screens]))
    report=dict(complete=True,paths=paths,screen=screens,audit=audits,
        oracle_mean_headroom=headroom,proceed=headroom>=.1,outer_predictions_opened=False,
        protocol_sha256=sha(H/'protocol.json'),source_hashes=sources,seconds=time.perf_counter()-start)
    save(O/'training.json',report)
    print(json.dumps(dict(training_complete=True,oracle_headroom=headroom,seconds=report['seconds'])),flush=True)

def outer_data(entry,sources):
    f,j=entry['outer'],entry['inner']; parts=[]
    for family in ['CNN','STT']:
        p=OUTER/family/f'fold{f}_inner{j}.npz'; proof=read(p.with_suffix('.json'))
        assert sha(p)==proof['sha256'] and sha(proof['checkpoint'])==proof['checkpoint_sha256']
        inner_proof=read((OLD/family/f'fold{f}_inner{j}.npz').with_suffix('.json'))
        assert inner_proof['checkpoint_sha256']==proof['checkpoint_sha256']
        identity=source_identity(inner_proof,sources)
        assert identity['train_people']==entry['base_people'] and identity['validation_people']==entry['meta_people']
        assert identity['outer_evaluation_people']==entry['outer_people']
        a=arrays(p); assert np.unique(a['person_id']).tolist()==entry['outer_people']
        assert (a['fold']==f).all(); parts.append(a);sources[str(p)]=sha(p)
    a,b=parts
    for key in a: 
        if key!='window_raw': np.testing.assert_array_equal(a[key],b[key])
    d=summarize(a['window_raw'],b['window_raw'],a['window_record'],a['row_index'],entry['coef_c'],entry['coef_s'])
    d.update(ids=a['person_id'],rows=a['row_index'],video=a['video_id'])
    return d

def metric_vectors(y,p,ids):
    return dict(mae=person_values(np.abs(p-y).mean(1),ids),
        mean_mae=person_values(np.abs(p.mean(1)-y.mean(1)),ids),
        side_difference_mae=person_values(np.abs((p[:,0]-p[:,1])-(y[:,0]-y[:,1])),ids),
        bias=person_values((p-y).mean(1),ids),underprediction=person_values(np.maximum(y-p,0).mean(1),ids))

def evaluate():
    start=time.perf_counter();cfg=read(H/'protocol.json');train=read(O/'training.json')
    assert train['complete'] and train['protocol_sha256']==sha(H/'protocol.json') and train['proceed']
    old=arrays(S/'mean-difference692/analysis/predictions.npz'); n=len(old['y']);sources={}
    names=['matched_fixed','shared','mean_difference','selected']
    out={k:np.full((n,2),np.nan) for k in names};allpaths={};gate_ranges=[]
    for f in range(5):
        chunks={k:[] for k in names}; rows=np.flatnonzero(old['fold']==f)
        for entry in [a for a in train['paths'] if a['outer']==f]:
            d=outer_data(entry,sources);np.testing.assert_array_equal(rows,d['rows'])
            np.testing.assert_array_equal(old['person_id'][rows],d['ids'])
            w=entry['fixed_weight']; pred={'matched_fixed':w*d['cnn']+(1-w)*d['stt']}
            for family in cfg['families']:
                pred[family],g=predict(entry['models'][family],d,cfg)
                gate_ranges.append(dict(outer=f,inner=entry['inner'],family=family,minimum=g.min(0).tolist(),maximum=g.max(0).tolist(),mean=g.mean(0).tolist()))
            family=entry['selected']['family'];pred['selected']=pred['matched_fixed' if family=='constant' else family]
            for k,v in pred.items():
                chunks[k].append(v);allpaths[f'f{f}_j{entry["inner"]}_{k}']=v
        for k in names: out[k][rows]=np.mean(chunks[k],axis=0)
    out['legacy_fixed']=old['existing_blend'];out['legacy_mean_difference']=old['inner_selected_difference']
    bag=read(S/'inner-bagging692/analysis/comparison.json'); assert sha(bag['predictions_path'])==bag['predictions_sha256']
    bz=arrays(bag['predictions_path'])
    for k in ['y','person_id','video_id','fold']: np.testing.assert_array_equal(old[k],bz[k])
    out['legacy_best']=bz['bagged']
    y=old['y'];ids=old['person_id'];people=np.unique(ids)
    high=np.array([y[ids==q].mean()>=90 for q in people]);assert high.sum()==68
    vectors={k:metric_vectors(y,v,ids) for k,v in out.items()}
    stats={k:{m:float(v.mean()) for m,v in a.items()} for k,a in vectors.items()}
    for k,a in vectors.items(): stats[k]['high90']={m:float(v[high].mean()) for m,v in a.items()};stats[k]['high90']['people']=int(high.sum())
    rng=np.random.default_rng(cfg['seed']);ix=rng.integers(0,len(people),(cfg['bootstrap_repeats'],len(people)))
    hi=rng.integers(0,high.sum(),(cfg['bootstrap_repeats'],high.sum()))
    paired={}
    for k in names[1:]:
        paired[k]={}
        for ref in ['matched_fixed','legacy_best']:
            r={}
            for m in vectors[k]:
                delta=vectors[k][m]-vectors[ref][m]
                r[m]=dict(delta=float(delta.mean()),bootstrap95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist())
                hd=delta[high];r['high90_'+m]=dict(delta=float(hd.mean()),bootstrap95=np.quantile(hd[hi].mean(1),[.025,.975]).tolist())
            paired[k][ref]=r
    fold_mae={k:[mae(y[old['fold']==f],p[old['fold']==f],ids[old['fold']==f]) for f in range(5)] for k,p in out.items()}
    report=dict(complete=True,people=len(people),records=n,metrics=stats,paired=paired,fold_mae=fold_mae,
        gate_ranges=gate_ranges,source_hashes=sources,training_sha256=sha(O/'training.json'),
        protocol_sha256=sha(H/'protocol.json'),protected_final_data_opened=False,seconds=time.perf_counter()-start)
    np.savez_compressed(O/'predictions.npz',**out,**{k:old[k] for k in ['y','person_id','video_id','fold']})
    np.savez_compressed(O/'path-predictions.npz',**allpaths)
    report['predictions_sha256']=sha(O/'predictions.npz');report['path_predictions_sha256']=sha(O/'path-predictions.npz')
    save(O/'comparison.json',report);print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);a=p.parse_args()
    train() if a.stage=='train' else evaluate()
