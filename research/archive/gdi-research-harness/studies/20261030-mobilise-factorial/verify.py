"""Independent input/scaler/replay/selection/refit and person-weighted metric checks."""
import json,itertools,time
import train as t
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
H=t.HERE

def independent_weights(d):
    counts=d.groupby('participantid').participantid.transform('size').to_numpy()
    w=1/counts;return w/w.sum()

def scales(tr,label,raw,k):
    w=independent_weights(tr);x=raw[tr.bag_id.to_numpy(),:k];y=tr[label].to_numpy();flat=x.reshape(-1,x.shape[-1]);fw=np.repeat(w/k,k)
    xm=np.average(flat,weights=fw,axis=0);xs=np.maximum(np.sqrt(np.average((flat-xm)**2,weights=fw,axis=0)),1e-8)
    ym=float(np.average(y,weights=w));ys=max(float(np.sqrt(np.average((y-ym)**2,weights=w))),1e-8)
    return x,y,w,{'x_mean':xm,'x_std':xs,'y_mean':np.array(ym),'y_std':np.array(ys)}

class IndependentNet(torch.nn.Module):
    def __init__(self,dim,width,maxpool):
        super().__init__();self.maxpool=maxpool;hidden=width//(4 if maxpool else 2)
        self.enc=torch.nn.Linear(dim,width);self.head=torch.nn.Linear(width*(2 if maxpool else 1),hidden);self.out=torch.nn.Linear(hidden,1)
    def forward(self,x):
        h=torch.relu(self.enc(x));z=h.mean(1)
        if self.maxpool:z=torch.cat((z,h.amax(1)),dim=1)
        return self.out(torch.relu(self.head(z))).squeeze(-1)

def independent_fit(tr,label,variant,width,epochs,seed,raw,k,valid=None):
    x,y,w,s=scales(tr,label,raw,k)
    xt=torch.tensor((x-s['x_mean'])/s['x_std'],dtype=torch.float32);yt=torch.tensor((y-float(s['y_mean']))/float(s['y_std']),dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
    torch.manual_seed(seed);net=IndependentNet(raw.shape[-1],width,variant in ['C','D']);optimizer=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=.001)
    curves=[]
    if valid is not None:
        vx=torch.tensor((raw[valid.bag_id.to_numpy(),:k]-s['x_mean'])/s['x_std'],dtype=torch.float32);vy=valid[label].to_numpy();vw=independent_weights(valid)
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();loss=(torch.nn.functional.smooth_l1_loss(net(xt),yt,beta=.5,reduction='none')*wt).sum();loss.backward();optimizer.step()
        if valid is not None and epoch%10==0:
            with torch.no_grad():vp=net(vx).numpy()*float(s['y_std'])+float(s['y_mean'])
            curves.append(float(np.dot(vw,abs(vp-vy))))
    for i,layer in enumerate([net.enc,net.head,net.out]):s['w'+str(i)]=layer.weight.detach().numpy();s['b'+str(i)]=layer.bias.detach().numpy()
    return s,curves

def restore(bundle,variant,width):
    net=t.make(variant,width,0)
    with torch.no_grad():
        for i,layer in enumerate([net.layers[0],net.layers[2],net.layers[4]]):layer.weight.copy_(torch.as_tensor(bundle['w'+str(i)]));layer.bias.copy_(torch.as_tensor(bundle['b'+str(i)]))
    return net

def numpy_predict(b,x,variant):
    z=((x-b['x_mean'])/b['x_std']).astype(np.float32);h=np.maximum(z@b['w0'].T+b['b0'],0);z=h.mean(1)
    if variant in ['C','D']:z=np.concatenate([z,h.max(1)],axis=1)
    y=np.maximum(z@b['w1'].T+b['b1'],0)@b['w2'].T+b['b2']
    return y[:,0]*float(b['y_std'])+float(b['y_mean'])

def verify_models():
    started=time.time();torch.set_num_threads(1);protocol=json.loads((H/'protocol.json').read_text());quality=json.loads((H/'quality.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert t.ref.sha(path)==digest,path
    for path,digest in quality['source_sha256'].items():assert t.ref.sha(path)==digest,path
    for path,digest in quality['artifact_sha256'].items():assert t.ref.sha(H/path)==digest,path
    assert t.ref.sha(H/'predictions.csv.gz')==json.loads((H/'completion.json').read_text())['predictions_sha256']
    data,arrays,ids=t.load_data();selected=pd.read_csv(t.BASE/'selected-bouts.csv').sort_values(t.ref.BAGKEY+['rank'])
    independent_core=selected[t.FEATURES].to_numpy().reshape(-1,5,4);np.testing.assert_array_equal(independent_core,arrays['A']);np.testing.assert_array_equal(arrays['A'],arrays['C'])
    independent_extra=np.stack([selected.duration.to_numpy(),(selected.turn_number_so/selected.duration).to_numpy()],axis=-1).reshape(-1,5,2)
    np.testing.assert_allclose(arrays['B'][:,:,4:],independent_extra,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(arrays['B'],arrays['D'])
    pd.testing.assert_frame_equal(ids[t.ref.BAGKEY].reset_index(drop=True),selected[t.ref.BAGKEY].drop_duplicates().reset_index(drop=True))
    p=pd.read_csv(H/'predictions.csv.gz').sort_values(t.KEY).reset_index(drop=True)
    assert np.isfinite(p[['actual','A','B','C','D']].to_numpy()).all()
    old=pd.read_csv(t.REFERENCE/'predictions.csv');old=old[old.budget.eq(600)].sort_values(t.KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(p[t.KEY],old[t.KEY]);np.testing.assert_array_equal(p.A.to_numpy(),old.prediction.to_numpy());np.testing.assert_array_equal(p.actual.to_numpy(),old.actual.to_numpy())
    assert len(p)==1173960 and not p.duplicated(t.KEY).any();assert set(p.seed)==set(t.SEEDS)
    model_count=0;selection_count=0;max_numpy_error=0.;max_permutation_error=0.;max_scale_error=0.
    refits={('sppb','short',1,0,t.SEEDS[0],'B'),('sppb','short',3,2,t.SEEDS[3],'C'),('sppb','long',5,4,t.SEEDS[4],'D'),('tug','short',5,1,t.SEEDS[1],'D'),('tug','long',1,2,t.SEEDS[2],'B'),('tug','long',5,3,t.SEEDS[3],'C')}
    inner_checks={('sppb','short',1,0,t.SEEDS[0],'B'):0,('sppb','long',5,4,t.SEEDS[4],'D'):1,('tug','long',5,3,t.SEEDS[3],'C'):2}
    baseline_inner_checks={('sppb','long',5,0,t.SEEDS[0]),('tug','long',5,2,t.SEEDS[3])}
    completed_refits=[];completed_inner=[]
    for target,band,k,fold,seed in itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),t.SEEDS):
        tag=f'{target}-{band}-{k}-f{fold}-s{seed}';label='total_sppb_score' if target=='sppb' else 'tugtime'
        cell=data[data[target+'_ok']&data.band.eq(band)&data.k.eq(k)].sort_values(['draw','participantid','visit_number'])
        tr=cell[cell.draw.eq(0)&cell.fold.ne(fold)].reset_index(drop=True);te=cell[cell.fold.eq(fold)].copy()
        ledger=json.loads((t.REFERENCE/'folds'/f'{tag}.json').read_text());width=ledger['selected']['600']['width']
        assert not set(tr.participantid)&set(te.participantid)
        assert set(tr.participantid)==set(ledger['train_people']) and set(te.participantid)==set(ledger['test_people'])
        cover=[]
        for split in ledger['inner_people']:
            assert not set(split['train'])&set(split['valid']);assert set(split['train'])|set(split['valid'])==set(tr.participantid);cover.extend(split['valid'])
        assert sorted(cover)==sorted(tr.participantid.unique())
        expected_best=min((r for r in ledger['trials'] if r['width']==width),key=lambda r:r['inner_person_mae']);assert expected_best==ledger['selected']['600']
        q=p[p.target.eq(target)&p.band.eq(band)&p.k.eq(k)&p.fold.eq(fold)&p.seed.eq(seed)].sort_values(['draw','participantid','visit_number'])
        pd.testing.assert_frame_equal(q[['participantid','visit_number','draw']].reset_index(drop=True),te[['participantid','visit_number','draw']].reset_index(drop=True));np.testing.assert_allclose(q.actual,te[label],atol=1e-12,rtol=0)
        for variant in t.VARIANTS:
            name=f'{tag}-{variant}'
            if variant=='A':modelpath=t.REFERENCE/'models'/f'{tag}-b600.npz';epochs=ledger['selected']['600']['epochs']
            else:
                record=json.loads((H/'folds'/f'{name}.json').read_text());assert record['reference_ledger_sha256']==t.ref.sha(t.REFERENCE/'folds'/f'{tag}.json');assert record['width']==width
                sizes=[len(s['valid']) for s in ledger['inner_people']];assert record['inner_people_counts']==sizes
                curve=np.average(record['inner_curves'],axis=0,weights=sizes);np.testing.assert_array_equal(curve,record['combined_curve']);choice=int(curve.argmin())
                assert record['epochs']==t.EPOCHS[choice] and record['inner_person_mae']==curve[choice];selection_count+=1;epochs=record['epochs'];modelpath=H/'models'/f'{name}.npz'
            b=dict(np.load(modelpath));_,_,_,sc=scales(tr,label,arrays[variant],k)
            for name_s in sc:
                np.testing.assert_allclose(b[name_s],sc[name_s],atol=1e-12,rtol=1e-12);max_scale_error=max(max_scale_error,float(np.max(abs(b[name_s]-sc[name_s]))))
            net=restore(b,variant,width);assert sum(v.numel() for v in net.parameters())==protocol['parameter_counts'][variant][str(width)]
            x=arrays[variant][te.bag_id.to_numpy(),:k];replay=t.ref.predict(net,b,x)
            np.testing.assert_array_equal(replay,q[variant].to_numpy(dtype=np.float32))
            n=numpy_predict(b,x,variant);np.testing.assert_allclose(n,replay,atol=2e-5,rtol=1e-5);max_numpy_error=max(max_numpy_error,float(np.max(abs(n-replay))))
            rev=t.ref.predict(net,b,x[:,::-1]);np.testing.assert_allclose(rev,replay,atol=2e-5,rtol=1e-5);max_permutation_error=max(max_permutation_error,float(np.max(abs(rev-replay))))
            model_count+=1
            task=(target,band,k,fold,seed,variant)
            if task in refits:
                fitted,_=independent_fit(tr,label,variant,width,epochs,seed+fold*10+9,arrays[variant],k)
                for field in b:np.testing.assert_array_equal(fitted[field],b[field])
                completed_refits.append(str(task))
            if task in inner_checks:
                inner=inner_checks[task];split=ledger['inner_people'][inner];it=tr[tr.participantid.isin(split['train'])];iv=tr[tr.participantid.isin(split['valid'])]
                _,curve=independent_fit(it,label,variant,width,600,seed+fold*10+inner,arrays[variant],k,iv)
                np.testing.assert_array_equal(curve,record['inner_curves'][inner]);completed_inner.append(str(task))
        if (target,band,k,fold,seed) in baseline_inner_checks:
            curves=[]
            for inner,split in enumerate(ledger['inner_people']):
                it=tr[tr.participantid.isin(split['train'])];iv=tr[tr.participantid.isin(split['valid'])]
                _,curve=independent_fit(it,label,'A',width,600,seed+fold*10+inner,arrays['A'],k,iv);curves.append(curve)
            average=np.average(curves,axis=0,weights=[len(s['valid']) for s in ledger['inner_people']])
            np.testing.assert_array_equal(average,[r['inner_person_mae'] for r in ledger['trials'] if r['width']==width]);completed_inner.append('reference:'+tag)
        if model_count%120==0:print('Verified',model_count,'of1200 model replays;',selection_count,'selections',flush=True)
    assert model_count==1200 and selection_count==900 and len(completed_refits)==6 and len(completed_inner)==5
    return {'model_replays_exact_float32':model_count,'numpy_and_permutation_replays':model_count,'max_numpy_absolute_difference':max_numpy_error,'max_permutation_absolute_difference':max_permutation_error,'max_scaler_absolute_difference':max_scale_error,'inner_selection_checks':selection_count,'independent_outer_refits_exact':completed_refits,'independent_inner_curves_exact':completed_inner,'predictions_rows_per_variant':len(p),'elapsed_seconds':time.time()-started}

def mask_for(d,group,target):
    tail=d.actual.le(3) if target=='sppb' else d.actual.gt(20)
    reviewed=d.sppb_component_review if target=='sppb' else d.tug_range_review
    if group=='all':return np.ones(len(d),dtype=bool)
    if group=='tail':return tail
    if group=='non_tail':return ~tail
    if group=='middle':return d.actual.gt(3)&d.actual.le(9)
    if group=='high':return d.actual.gt(9)
    if group.startswith('cohort_'):return d.cohort.eq(group[7:])
    if group=='unflagged_all':return ~reviewed
    if group=='unflagged_tail':return ~reviewed&tail
    if group=='turn_rate_below1_all':return d.max_turn_rate.lt(1)
    if group=='turn_rate_below1_tail':return d.max_turn_rate.lt(1)&tail
    raise ValueError(group)

def verify_metrics():
    import analyze as a
    p=a.load();m=pd.read_csv(H/'seed-metrics.csv');keys=['target','band','k','draw','seed'];checked=0
    # Verify metadata used by sensitivity analyses directly against the released labels.
    source=t.ref.ROOT/'outputs/sppb-tug-candidates-20261030/All_cohorts_dataset.csv'
    labels=pd.read_csv(source,usecols=['participantid','visit.number','tugtime','total_sppb_score','balance_score','4mwt_speed_score','5crt_score'],low_memory=False).rename(columns={'visit.number':'visit_number'})
    labels=labels.merge(pd.read_csv(H/'label-review-flags.csv'),on=['participantid','visit_number'],validate='one_to_one')
    cols=['balance_score','4mwt_speed_score','5crt_score']
    np.testing.assert_array_equal(labels.sppb_component_review,(labels[cols].isna().any(axis=1)|labels[cols].sum(axis=1).ne(labels.total_sppb_score)))
    np.testing.assert_array_equal(labels.tug_range_review,(labels.tugtime.lt(3)|labels.tugtime.gt(60)))
    for key,d in p.groupby(keys):
        rows=m
        for field,value in zip(keys,key):rows=rows[rows[field].eq(value)]
        for group,rs in rows.groupby('group'):
            g=d[mask_for(d,group,key[0])]
            for r in rs.itertuples(index=False):
                error=g[r.variant]-g.actual;absolute=error.abs();person=error.groupby(g.participantid)
                mae=absolute.groupby(g.participantid).mean().mean();bias=person.mean().mean();rmse=np.sqrt(error.pow(2).groupby(g.participantid).mean().mean())
                np.testing.assert_allclose([r.mae,r.bias,r.rmse],[mae,bias,rmse],rtol=1e-12,atol=1e-12)
                w=1/g.groupby('participantid').participantid.transform('size')/g.participantid.nunique()
                assert float(w[absolute.lt(r.error_p90-1e-12)].sum())<=.9+1e-12
                assert float(w[absolute.le(r.error_p90+1e-12)].sum())>=.9-1e-12
                assert float(abs(absolute-r.error_p90).min())<1e-10
                assert r.people==g.participantid.nunique() and r.visits==len(g);checked+=1
    assert checked==len(m)==8640
    previous=pd.read_csv(t.REFERENCE/'seed-metrics.csv');previous=previous[previous.budget.eq(600)].drop(columns='budget')
    previous['group']=previous.group.replace({'low':'tail','long':'tail','other':'non_tail'})
    current=m[m.variant.eq('A')].drop(columns='variant').merge(previous,on=keys+['group'],suffixes=('','_reference'),validate='one_to_one')
    assert len(current)==len(previous)
    for field in ['mae','bias','rmse','error_p90']:np.testing.assert_allclose(current[field],current[field+'_reference'],atol=1e-10,rtol=1e-10)
    coefficients={'input_mean':[-1,1,0,0],'input_meanmax':[0,0,-1,1],'structure_core':[-1,0,1,0],'structure_expanded':[0,-1,0,1],'combined':[-1,0,0,1],'interaction':[1,-1,-1,1],'input_average':[-.5,.5,-.5,.5],'structure_average':[-.5,-.5,.5,.5]}
    effects=pd.read_csv(H/'seed-factorial-effects.csv');ef=effects.set_index(keys+['group','contrast'])
    for key,d in m.groupby(keys+['group']):
        values=d.set_index('variant').loc[['A','B','C','D']]
        for contrast,c in coefficients.items():
            row=ef.loc[key+(contrast,)]
            for field in ['mae','bias','rmse','error_p90']:np.testing.assert_allclose(row[field+'_change'],sum(float(c[i])*float(values.iloc[i][field]) for i in range(4)),atol=1e-12,rtol=1e-12)
    agg=pd.read_csv(H/'aggregate-metrics.csv');groups=['target','band','k','draw','variant','group']
    for field in ['mae','bias','rmse','error_p90']:
        calculated=m.groupby(groups)[field].agg(['mean','std','min','max']).reset_index().merge(agg,on=groups,validate='one_to_one')
        for stat,column in [('mean','mean'),('std','sd'),('min','min'),('max','max')]:np.testing.assert_allclose(calculated[stat],calculated[field+'_'+column],atol=1e-12,rtol=1e-12)
    trade=pd.read_csv(H/'seed-tradeoffs.csv')
    np.testing.assert_array_equal(trade.overall_not_worse,trade.mae_change_all.le(0))
    np.testing.assert_array_equal(trade.tail_better_global_not_worse,trade.mae_change_all.le(0)&trade.mae_change_tail.lt(0))
    np.testing.assert_array_equal(trade.tail_mae_and_p90_better_global_not_worse,trade.mae_change_all.le(0)&trade.mae_change_tail.lt(0)&trade.error_p90_change_tail.lt(0))
    conditions=pd.read_csv(H/'condition-tradeoffs.csv');group_keys=['target','band','k','draw','contrast','view'];condition_checked=0
    for key,d in trade.groupby(group_keys):
        r=conditions
        for field,value in zip(group_keys,key):r=r[r[field].eq(value)]
        assert len(r)==1 and len(d)==5;r=r.iloc[0]
        np.testing.assert_allclose([r.global_mae_change,r.tail_mae_change,r.tail_p90_change,r.tail_rmse_change],[d.mae_change_all.mean(),d.mae_change_tail.mean(),d.error_p90_change_tail.mean(),d.rmse_change_tail.mean()],atol=1e-12,rtol=1e-12)
        assert r.global_not_worse_seeds==int(d.overall_not_worse.sum()) and r.tail_better_global_not_worse_seeds==int(d.tail_better_global_not_worse.sum()) and r.tail_mae_p90_better_global_not_worse_seeds==int(d.tail_mae_and_p90_better_global_not_worse.sum());condition_checked+=1
    intervals=pd.DataFrame(json.loads((H/'paired-bootstrap.json').read_text())).set_index(['target','band','k','group','contrast']);rng=np.random.default_rng(20261030);boot_count=0
    for (target,band,k),d in p[p.draw.eq(0)].groupby(['target','band','k']):
        for group in ['all','tail','unflagged_all','unflagged_tail']:
            g=d[mask_for(d,group,target)];columns=[]
            for variant in ['A','B','C','D']:columns.append((g[variant]-g.actual).abs().groupby(g.participantid).mean())
            people=pd.concat(columns,axis=1);boot=rng.integers(0,len(people),size=(2000,len(people)))
            for contrast,c in coefficients.items():
                delta=(people.to_numpy()*np.array(c)).sum(axis=1);lo,hi=np.quantile(delta[boot].mean(axis=1),[.025,.975]);r=intervals.loc[(target,band,k,group,contrast)]
                np.testing.assert_allclose([r.mae_change,r.ci_low,r.ci_high],[delta.mean(),lo,hi],atol=1e-12,rtol=1e-12)
                assert r.people==len(people) and r.people_improved==int((delta<0).sum()) and r.people_worsened==int((delta>0).sum());boot_count+=1
    return {'independent_seed_metrics':checked,'reference_metrics_unchanged':len(current),'factorial_effects_checked':len(effects),'condition_tradeoffs_checked':condition_checked,'independent_bootstrap_intervals':boot_count}

def run():
    started=time.time();models=verify_models();print('All model/selection/refit checks passed',flush=True);metrics=verify_metrics();print('All metric/bootstrap checks passed',flush=True)
    summary=json.loads((H/'summary.json').read_text())
    for name,digest in summary['artifact_sha256'].items():assert t.ref.sha(H/name)==digest
    manifest={str(path.relative_to(H)):t.ref.sha(path) for folder,pattern in [('models','*.npz'),('folds','*.json'),('folds','*.csv.gz')] for path in sorted((H/folder).glob(pattern))}
    assert len(manifest)==2700;t.save('model-manifest.json',manifest)
    names=['protocol.json','quality.json','quality-context.json','preflight.json','completion.json','predictions.csv.gz','summary.json','model-manifest.json','audited-bouts.csv','label-review-flags.csv','turn-distribution.json','factorial-effects.png']+list(summary['artifact_sha256'])
    t.save('verification.json',{'passed':True,**models,**metrics,'elapsed_seconds_total':time.time()-started,'artifact_sha256':{name:t.ref.sha(H/name) for name in names},'scripts_sha256':{name:t.ref.sha(H/name) for name in ['quality.py','train.py','analyze.py','verify.py','plot.py']}})
    print(json.dumps(json.loads((H/'verification.json').read_text()),indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
