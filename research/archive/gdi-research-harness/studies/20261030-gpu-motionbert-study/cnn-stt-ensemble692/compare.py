"""Separate selection (inner only) and evaluation stages; immutable base runs."""
from pathlib import Path
import argparse, hashlib, json, sys, tarfile
import numpy as np

H=Path(__file__).resolve().parent
S=H.parent
C=S/'cnn-multiwindow695'
T=S/'stt692/multiwindow'
O=H/'analysis'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def errors(y,p,ids):
    assert y.shape==p.shape==(len(ids),2) and np.isfinite(p).all()
    return np.array([np.abs(p[ids==q]-y[ids==q]).mean() for q in np.unique(ids)])
def choose(cnn,stt,y,ids,grid):
    scores=[float(errors(y,w*cnn+(1-w)*stt,ids).mean()) for w in grid]
    i=min(range(len(grid)),key=lambda k:(scores[k],abs(grid[k]-.5),grid[k]))
    return grid[i],scores

def extract():
    archive=H/'received/gdi-cnn-inner-for-blend-20261031.tar.gz'
    assert sha(archive)=='bfd8b0a3245c79e7a750915dfa52b75e329f07990ed52b53e4718d7ab33b0680'
    root=H/'received/checkpoints';root.mkdir(exist_ok=True)
    with tarfile.open(archive) as tf:
        members=tf.getmembers();assert len(members)==20
        for m in members:
            assert m.isfile() and len(Path(m.name).parts)==3 and m.name.endswith('/latest.pt')
            dest=(root/m.name).resolve();assert root.resolve() in dest.parents
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(m).read())
    return root

def select():
    # This stage does not open any outer prediction file or protected final data.
    sys.path.insert(0,str(C/'bundle'))
    import torch,run as r
    r.setup(20261031,'cpu');d=r.data();cfg=read(H/'protocol.json');root=extract()
    O.mkdir(exist_ok=True);source_hashes={};selections=[]
    def hashed(p):source_hashes[str(p.relative_to(S))]=sha(p)
    for f in range(5):
        rows=[];cp=[];sp=[];audits=[]
        for j in range(5):
            if j==f:continue
            tr,va,te=r.split(d,f,j);ci=C/'received/results'/('fold%d/inner%d'%(f,j))
            si=T/'analysis/unpacked/results'/('fold%d/inner%d'%(f,j))
            ckpath=root/('fold%d/inner%d/latest.pt'%(f,j))
            complete=read(ci/'fit_complete.json');ident=read(ci/'identity.json')
            assert sha(ckpath)==complete['checkpoint_sha256']
            ck=torch.load(ckpath,map_location='cpu',weights_only=True)
            assert ck['identity']==ident==complete['identity'] and len(ck['history'])==cfg['cnn_inner_epoch']
            for directory in (ci,si):
                ii=read(directory/'identity.json')
                for key,mask in [('train_people',tr),('validation_people',va),('outer_evaluation_people',te)]:
                    assert ii[key]==np.unique(d['person_id'][mask]).tolist()
                assert ii['outer_fold']==f and ii['inner_fold']==j
                assert not set(ii['train_people']) & set(ii['validation_people'])
                assert not (set(ii['train_people'])|set(ii['validation_people'])) & set(ii['outer_evaluation_people'])
            lo,span=r.scale_from_train(d['y'][tr])
            assert torch.equal(ck['lo'],lo) and torch.equal(ck['span'],span)
            model=r.GdiCNN(r.config()['dropout']);model.load_state_dict(ck['model']);model.eval()
            raw_train=r.predict(model,d['x'][tr],lo,span,'cpu')
            raw_valid=r.predict(model,d['x'][va],lo,span,'cpu')
            coef=r.calibrate(raw_train,d['y'][tr]);stored=np.array(ck['history'][-1]['calibration'])
            # Use source-GPU calibration after checking CPU replay at existing 1e-4 tolerance.
            calibration_delta=float(np.max(abs(r.corrected(raw_valid,coef)-r.corrected(raw_valid,stored))))
            assert calibration_delta<1e-4
            cnn=r.corrected(raw_valid,stored)
            metric_delta=abs(r.person_mae(d['y'][va],cnn,d['person_id'][va])-ck['history'][-1]['validation_person_mae'])
            assert metric_delta<1e-4
            h=read(si/'curve.json');done=read(si/'fit_complete.json');last=h[-1]
            assert len(h)==cfg['stt_inner_epoch'] and sha(si/'curve.json')==done['curve_sha256']
            assert sha(si/'latest.pt')==done['checkpoint_sha256']
            rawfile=si/'raw_predictions'/('epoch%04d.npz'%cfg['stt_inner_epoch'])
            assert sha(rawfile)==last['raw_predictions_sha256']
            zz=arrays(rawfile);sc=r.calibrate(zz['train_raw'],d['y'][tr])
            np.testing.assert_allclose(sc,last['calibration'],rtol=0,atol=1e-10)
            stt=r.corrected(zz['validation_raw'],last['calibration'])
            assert abs(r.person_mae(d['y'][va],stt,d['person_id'][va])-last['validation_person_mae'])<1e-10
            ix=np.flatnonzero(va);rows.extend(ix);cp.extend(cnn);sp.extend(stt)
            for path in (ckpath,ci/'identity.json',ci/'fit_complete.json',si/'identity.json',si/'fit_complete.json',si/'curve.json',si/'latest.pt',rawfile):hashed(path)
            audits.append(dict(inner=j,cnn_validation_metric_replay_delta=metric_delta,cnn_calibration_prediction_delta=calibration_delta,train_people=int(len(np.unique(d['person_id'][tr]))),validation_people=int(len(np.unique(d['person_id'][va])))))
            print(json.dumps(dict(exported_outer=f,inner=j)),flush=True)
        rows=np.array(rows);order=np.argsort(rows);rows=rows[order];cnn=np.asarray(cp)[order];stt=np.asarray(sp)[order]
        np.testing.assert_array_equal(rows,np.flatnonzero(d['fold']!=f))
        assert len(np.unique(rows))==len(rows)
        ids=d['person_id'][rows];y=d['y'][rows]
        weight,scores=choose(cnn,stt,y,ids,cfg['cnn_weight_grid'])
        # Independent weighted-vector calculation gives each person equal total mass.
        ew=np.array([1/np.sum(ids==q) for q in ids]);ew=ew/ew.sum()
        independent=[float(np.sum(np.abs(w*cnn+(1-w)*stt-y).mean(1)*ew)) for w in cfg['cnn_weight_grid']]
        np.testing.assert_allclose(scores,independent,rtol=0,atol=1e-12)
        np.savez_compressed(O/('inner_outer%d.npz'%f),cnn=cnn,stt=stt,y=y,person_id=ids,video_id=d['video_id'][rows],fold=d['fold'][rows],row_index=rows)
        selections.append(dict(outer=f,cnn_weight=weight,stt_weight=1-weight,inner_person_mae=scores,inner_people=len(np.unique(ids)),outer_evaluation_used=False,audits=audits,inner_file_sha256=sha(O/('inner_outer%d.npz'%f))))
    save(O/'selection.json',dict(protocol_sha256=sha(H/'protocol.json'),folds=selections,source_hashes=source_hashes,outer_predictions_opened=False,neural_optimizer_steps=0))
    print(json.dumps({'selection_complete':True,'cnn_weights':[s['cnn_weight'] for s in selections]}))

def evaluate():
    cfg=read(H/'protocol.json');selection=read(O/'selection.json')
    assert selection['protocol_sha256']==sha(H/'protocol.json')
    z=arrays(C/'analysis/cnn_oof.npz');v=arrays(T/'analysis/unpacked/results/oof_predictions.npz')
    assert len(np.unique(z['person_id']))==692 and len(z['y'])==1061
    lookup={q:i for i,q in enumerate(v['video_id'])};assert len(lookup)==1061
    ix=np.array([lookup[q] for q in z['video_id']])
    for k in ('person_id','y','fold'):np.testing.assert_array_equal(z[k],v[k][ix])
    cnn=z['predictions'][:,0];stt=v['predictions'][ix,0];tuned=np.full_like(cnn,np.nan)
    for sf in selection['folds']:
        f=sf['outer'];inner=arrays(O/('inner_outer%d.npz'%f))
        assert sha(O/('inner_outer%d.npz'%f))==sf['inner_file_sha256']
        assert not (set(inner['person_id']) & set(z['person_id'][z['fold']==f]))
        w,score=choose(inner['cnn'],inner['stt'],inner['y'],inner['person_id'],cfg['cnn_weight_grid'])
        assert w==sf['cnn_weight'];np.testing.assert_array_equal(score,sf['inner_person_mae'])
        mask=z['fold']==f;tuned[mask]=w*cnn[mask]+(1-w)*stt[mask]
    models=dict(CNN=cnn,STT=stt,fixed_half=.5*cnn+.5*stt,nested_weight=tuned)
    es={k:errors(z['y'],p,z['person_id']) for k,p in models.items()}
    comparisons={}
    for name in ('fixed_half','nested_weight'):
        delta=es[name]-es['CNN'];rng=np.random.default_rng(20261031)
        boot=np.array([delta[rng.integers(0,692,692)].mean() for _ in range(10000)])
        comparisons[name]=dict(delta_vs_CNN=float(delta.mean()),better_people=int((delta<0).sum()),worse_people=int((delta>0).sum()),paired_person_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    result=dict(person_equal_mae={k:float(e.mean()) for k,e in es.items()},fold_mae={k:[float(errors(z['y'][z['fold']==f],p[z['fold']==f],z['person_id'][z['fold']==f]).mean()) for f in range(5)] for k,p in models.items()},cnn_weights=[v['cnn_weight'] for v in selection['folds']],paired=comparisons,people=692,records=1061,selection_sha256=sha(O/'selection.json'),outer_source_sha256={'CNN':sha(C/'analysis/cnn_oof.npz'),'STT':sha(T/'analysis/unpacked/results/oof_predictions.npz')},independent_final_confirmation=False,protected_final_data_opened=False,neural_optimizer_steps=0,limitations=cfg['limitations'])
    np.savez_compressed(O/'predictions.npz',**models,**{k:z[k] for k in ('y','person_id','video_id','fold')})
    save(O/'comparison.json',result);print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','evaluate'])
    args=parser.parse_args();select() if args.stage=='select' else evaluate()
