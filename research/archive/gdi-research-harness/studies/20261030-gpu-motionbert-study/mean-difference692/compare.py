"""Selection never opens outer predictions. All derived outputs are small C files."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
H=Path(__file__).resolve().parent
S=H.parent
E=S/'cnn-stt-ensemble692/analysis'
B=S/'stt692/bilateral/analysis/unpacked/results'
O=H/'analysis'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def errors(y,p,ids):
    assert y.shape==p.shape==(len(ids),2) and np.isfinite(p).all()
    return np.array([np.abs(y[ids==i]-p[ids==i]).mean() for i in np.unique(ids)])
def combine(base,bilateral,fraction):
    common=base.mean(axis=1)
    delta=(1-fraction)*(base[:,0]-base[:,1])+fraction*(bilateral[:,0]-bilateral[:,1])
    return np.column_stack([common+delta/2,common-delta/2])
def choose(base,bilateral,y,ids,grid):
    scores=[float(errors(y,combine(base,bilateral,t),ids).mean()) for t in grid]
    k=min(range(len(grid)),key=lambda k:(scores[k],grid[k]))
    return grid[k],scores
def select():
    cfg=read(H/'protocol.json');old=read(E/'selection.json');proof=read(E/'verification.json')
    assert proof['passed'] and proof['all_20_base_fit_person_boundaries']
    assert proof['comparison_sha256']==sha(E/'comparison.json')
    assert old['protocol_sha256']==sha(E.parent/'protocol.json')
    hashes={str(E/'selection.json'):sha(E/'selection.json'),str(E/'verification.json'):sha(E/'verification.json')}
    # Recheck the frozen evidence supporting the reusable CNN/STT inner cache.
    for p,digest in old['source_hashes'].items():assert sha(S/p)==digest
    with np.load(S/'stt692/bilateral/bundle/data.npz',allow_pickle=False) as z:
        d={k:z[k].copy() for k in ('y','person_id','video_id','fold')}
    selections=[]
    for f in range(5):
        innerpath=E/('inner_outer%d.npz'%f);inner=arrays(innerpath);sf=old['folds'][f]
        assert sf['outer']==f and sha(innerpath)==sf['inner_file_sha256']
        ix=inner['row_index'];np.testing.assert_array_equal(ix,np.flatnonzero(d['fold']!=f))
        for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(inner[key],d[key][ix])
        bilateral=np.full_like(inner['cnn'],np.nan)
        for j in range(5):
            if f==j:continue
            folder=B/('fold%d/inner%d'%(f,j));ident=read(folder/'identity.json');done=read(folder/'fit_complete.json')
            curves=read(folder/'curve.json');assert len(curves)==cfg['inner_epochs']['bilateral_STT']
            assert done['identity']==ident and done['curve_sha256']==sha(folder/'curve.json')
            tr=(d['fold']!=f)&(d['fold']!=j);va=d['fold']==j;te=d['fold']==f
            for key,mask in [('train_people',tr),('validation_people',va),('outer_evaluation_people',te)]:
                assert ident[key]==np.unique(d['person_id'][mask]).tolist()
            assert not set(ident['train_people'])&set(ident['validation_people'])
            assert not (set(ident['train_people'])|set(ident['validation_people']))&set(ident['outer_evaluation_people'])
            path=folder/'raw_predictions/epoch0200.npz';assert sha(path)==curves[-1]['raw_predictions_sha256']
            z=arrays(path);coef=np.linalg.lstsq(np.column_stack([z['train_raw'].ravel(),np.ones(z['train_raw'].size)]),d['y'][tr].ravel(),rcond=None)[0]
            np.testing.assert_allclose(coef,curves[-1]['calibration'],atol=1e-10,rtol=0)
            pred=z['validation_raw']*coef[0]+coef[1]
            assert abs(errors(d['y'][va],pred,d['person_id'][va]).mean()-curves[-1]['validation_person_mae'])<1e-10
            pos=inner['fold']==j;np.testing.assert_array_equal(ix[pos],np.flatnonzero(va));bilateral[pos]=pred
            for p in (folder/'identity.json',folder/'fit_complete.json',folder/'curve.json',path):hashes[str(p)]=sha(p)
        base=sf['cnn_weight']*inner['cnn']+(1-sf['cnn_weight'])*inner['stt']
        t,scores=choose(base,bilateral,inner['y'],inner['person_id'],cfg['difference_fraction_grid'])
        dest=O/('inner_outer%d.npz'%f);O.mkdir(exist_ok=True)
        np.savez_compressed(dest,base=base,bilateral=bilateral,**{k:inner[k] for k in ('y','person_id','video_id','fold','row_index')})
        selections.append(dict(outer=f,cnn_weight=sf['cnn_weight'],fraction=t,scores=scores,inner_sha256=sha(dest),outer_evaluation_used=False))
        hashes[str(innerpath)]=sha(innerpath)
    save(O/'selection.json',dict(protocol_sha256=sha(H/'protocol.json'),folds=selections,source_hashes=hashes,outer_predictions_opened=False,neural_optimizer_steps=0))
    print(json.dumps({'selected_fractions':[x['fraction'] for x in selections]}),flush=True)
def evaluate():
    cfg=read(H/'protocol.json');sel=read(O/'selection.json');assert sel['protocol_sha256']==sha(H/'protocol.json')
    z=arrays(E/'predictions.npz');b=arrays(B/'oof_predictions.npz');index={v:i for i,v in enumerate(b['video_id'])}
    assert len(index)==1061;ix=np.array([index[v] for v in z['video_id']])
    for key in ('y','person_id','fold'):np.testing.assert_array_equal(z[key],b[key][ix])
    base=z['nested_weight'];bp=b['predictions'][ix,0];fixed=combine(base,bp,1);tuned=np.full_like(base,np.nan)
    for sf in sel['folds']:
        f=sf['outer'];inner=arrays(O/('inner_outer%d.npz'%f));assert sha(O/('inner_outer%d.npz'%f))==sf['inner_sha256']
        mask=z['fold']==f;assert not set(inner['person_id'])&set(z['person_id'][mask])
        t,score=choose(inner['base'],inner['bilateral'],inner['y'],inner['person_id'],cfg['difference_fraction_grid'])
        assert t==sf['fraction'];np.testing.assert_allclose(score,sf['scores'],atol=1e-12,rtol=0)
        tuned[mask]=combine(base[mask],bp[mask],t)
    models={'CNN':z['CNN'],'STT':z['STT'],'existing_blend':base,'bilateral_STT':bp,'full_difference_replacement':fixed,'inner_selected_difference':tuned}
    ids=z['person_id'];y=z['y'];es={k:errors(y,p,ids) for k,p in models.items()};people=np.unique(ids)
    report={'mae':{k:float(v.mean()) for k,v in es.items()},'fold_mae':{},'components':{},'high90':{},'paired_vs_existing_blend':{},'fractions':[x['fraction'] for x in sel['folds']]}
    high=np.array([y[ids==i].mean()>=90 for i in people])
    for k,p in models.items():
        report['fold_mae'][k]=[float(errors(y[z['fold']==f],p[z['fold']==f],ids[z['fold']==f]).mean()) for f in range(5)]
        report['components'][k]={'common_mae':float(np.mean([np.abs(p[ids==i].mean(1)-y[ids==i].mean(1)).mean() for i in people])), 'signed_difference_mae':float(np.mean([np.abs(np.diff(p[ids==i],axis=1)-np.diff(y[ids==i],axis=1)).mean() for i in people]))}
        report['high90'][k]={'people':int(high.sum()),'mae':float(es[k][high].mean()),'band_definition':'person mean actual GDI >=90'}
    for k in ('full_difference_replacement','inner_selected_difference'):
        delta=es[k]-es['existing_blend'];rng=np.random.default_rng(20261031)
        boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(10000)])
        report['paired_vs_existing_blend'][k]={'delta':float(delta.mean()),'better_people':int((delta<0).sum()),'worse_people':int((delta>0).sum()),'bootstrap95_descriptive':np.quantile(boot,[.025,.975]).tolist()}
        np.testing.assert_allclose(models[k].mean(1),base.mean(1),atol=1e-12,rtol=0)
    np.savez_compressed(O/'predictions.npz',**models,**{k:z[k] for k in ('y','person_id','video_id','fold')})
    report.update(people=692,records=1061,protected_final_data_opened=False,neural_optimizer_steps=0,selection_sha256=sha(O/'selection.json'),outer_sources={str(p):sha(p) for p in (E/'predictions.npz',B/'oof_predictions.npz')},limitations=cfg['limitations'])
    save(O/'comparison.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','evaluate']);a=p.parse_args()
    select() if a.stage=='select' else evaluate()
