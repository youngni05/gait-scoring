"""Independent CPU replay of every returned checkpoint and nested selection."""
import argparse,csv
import numpy as np
import torch
from common import load,save,sha,data,mask,check_plan,person_rows
from model import TransferCNN
from bn_verify import verify_bn

def check_fit(root,p,d,tr,va,test,channels,seed):
    ident=load(p/'identity.json');result=load(p/'fit_complete.json');curve=load(p/'curve.json')
    expected=dict(lr=.001,weight_decay=.001,dropout=.5,freeze_blocks=0,loss='mse',channels=channels)
    assert ident['config']==expected and ident['seed']==seed and ident['source']=='scratch' and ident['initialization_sha256'] is None
    assert ident['train_people']==sorted(tr) and ident['validation_people']==sorted(va) and ident['test_people']==sorted(test)
    assert not set(tr)&(set(va)|set(test))
    assert ident['input_sha256']==sha(root/'target.npz') and ident['plan_sha256']==sha(root/'plan.json')
    assert result['checkpoint_sha256']==sha(p/'model.pt')
    assert len(curve)==result['epochs_run']
    assert [c['epoch'] for c in curve]==list(range(1,len(curve)+1))
    if va:
        best=min(curve,key=lambda c:(c['validation_MAE'],c['epoch']))
        assert best['epoch']==result['best_epoch'] and abs(best['validation_MAE']-result['best_validation_MAE'])<1e-10
    else:
        assert result['best_epoch']==len(curve)==ident['max_epochs'] and result['best_validation_MAE'] is None
    tw=int(mask(d,tr)[d['owner']].sum());assert result['optimizer_steps']==int(np.ceil(tw/32))*len(curve)
    bn=verify_bn(p,d,mask(d,tr),sha(root/'target.npz'))
    ck=torch.load(p/'model.pt',map_location='cpu',weights_only=True)
    yy=np.array([d['y'][d['ids']==pid].mean() for pid in sorted(tr)],np.float64)
    np.testing.assert_allclose(ck['mu'].numpy(),[yy.mean()],atol=1e-6,rtol=0)
    np.testing.assert_allclose(ck['sd'].numpy(),[max(yy.std(),1e-4)],atol=1e-6,rtol=0)
    model=TransferCNN(1,.5,channels);model.load_state_dict(ck['model']);model.eval()
    held=va if va else test;wi=np.flatnonzero(mask(d,held)[d['owner']]);pred=np.zeros((len(d['ids']),1),np.float64)
    with torch.no_grad():
        for a in range(0,len(wi),128):
            q=wi[a:a+128];pred[d['owner'][q]]=model(torch.from_numpy(d['x'][q])).numpy()*ck['sd'].numpy()+ck['mu'].numpy()
    rows=person_rows(d,pred,held)
    if va:assert abs(np.mean([abs(r['prediction']-r['y']) for r in rows])-result['best_validation_MAE'])<1e-4
    return result,rows,bn
def compare_rows(actual,recorded):
    assert [r['id'] for r in actual]==[r['id'] for r in recorded]
    np.testing.assert_array_equal([r['y'] for r in actual],[r['y'] for r in recorded])
    a=np.array([r['prediction'] for r in actual]);b=np.array([r['prediction'] for r in recorded])
    np.testing.assert_allclose(a,b,atol=1e-4,rtol=0)
    return float(np.max(np.abs(a-b)))
def verify(root,out,pilot=False):
    torch.set_num_threads(2)
    for n,h in load(root/'manifest.json').items():assert sha(root/n)==h,n
    d=data(root/'target.npz');plan=load(root/'plan.json');check_plan(plan,d)
    rows=[];counts=0;bn_error=0.;prediction_error=0.;selections=[]
    if pilot:
        summary=load(root/'runs/pilot/summary.json');assert summary['passed'] and summary['cuda']
        assert len(summary['fits'])==6
        combos=set()
        for fit in summary['fits']:
            case=next(c for c in plan['cases'] if c['fold']==0 and c['key']==fit['case']);split=case['inner'][0]
            assert (case['size']==8 and case['draw']==0) or case['size']=='full'
            p=root/fit['path'];r,actual,bn=check_fit(root,p,d,split['train'],split['validation'],case['test'],fit['channels'],20261101)
            bn_error=max(bn_error,bn);prediction_error=max(prediction_error,compare_rows(actual,load(p/'validation-predictions.json')))
            ck=torch.load(p/'model.pt',map_location='cpu',weights_only=True);model=TransferCNN(1,.5,fit['channels']);model.load_state_dict(ck['model']);model.eval()
            probe=torch.from_numpy(d['x'][:5])
            with torch.no_grad():np.testing.assert_allclose(model(probe).numpy(),torch.cat([model(x[None]) for x in probe]).numpy(),atol=1e-5,rtol=0)
            combos.add((case['key'],fit['channels']));counts+=1
        assert len(combos)==6
    else:
        summary=load(root/'runs/round0/summary.json');assert summary['total_fits']==1800
        for case in plan['cases']:
            base=root/('runs/round0/fold%d'%case['fold'])/case['key']
            assert load(base.parent/'complete.json')['total_fits']==360
            for ch in (32,16,8):
                for seed in (20261101,20261102,20261103):
                    arm=base/('cnn%d-s%d'%(ch,seed));inner=[]
                    for j,split in enumerate(case['inner']):
                        r,_,bn=check_fit(root,arm/('inner%d'%j),d,split['train'],split['validation'],case['test'],ch,seed+case['fold']*100+j)
                        inner.append(r);bn_error=max(bn_error,bn);counts+=1
                    sel=load(arm/'selection.json');epochs=int(np.median([r['best_epoch'] for r in inner]))
                    score=float(np.average([r['best_validation_MAE'] for r in inner],weights=[len(s['validation']) for s in case['inner']]))
                    assert sel['epochs']==epochs and abs(sel['inner_MAE']-score)<1e-10
                    r,actual,bn=check_fit(root,arm/'refit',d,case['people'],[],case['test'],ch,seed+case['fold']*100+50)
                    assert r['best_epoch']==epochs and load(arm/'complete.json')['channels']==ch
                    prediction_error=max(prediction_error,compare_rows(actual,load(arm/'predictions.json')));bn_error=max(bn_error,bn);counts+=1
                    rows += [dict(q,fold=case['fold'],case=case['key'],size=case['size'],draw=case['draw'],model='CNN%d'%ch,seed=seed) for q in load(arm/'predictions.json')]
                    selections.append(dict(fold=case['fold'],case=case['key'],channels=ch,seed=seed,epochs=epochs))
            print('Verified case %d/%d; fits %d'%(plan['cases'].index(case)+1,len(plan['cases']),counts),flush=True)
        assert counts==1800 and len(rows)==3960
    out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (out/'person-predictions.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        save(out/'selection.json',selections)
    result=dict(passed=True,pilot=pilot,people=44,verified_fits=counts,final_models=0 if pilot else 450,max_cpu_prediction_error=prediction_error,max_bn_buffer_error=bn_error,input_sha256=sha(root/'target.npz'),plan_sha256=sha(root/'plan.json'))
    save(out/'verification.json',result);print(result,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=__import__('pathlib').Path,required=True);ap.add_argument('--out',type=__import__('pathlib').Path,required=True);ap.add_argument('--pilot',action='store_true');a=ap.parse_args();verify(a.root,a.out,a.pilot)
