"""Three channel counts, nested participant sampling and fixed outer evaluation."""
import os,argparse,socket,platform
import numpy as np
import torch
from common import ROOT,load,save,sha,data,mask,people,check_plan,person_rows
from fit import fit,predict,setup
from model import TransferCNN

def config(channels):return dict(lr=.001,weight_decay=.001,dropout=.5,freeze_blocks=0,loss='mse',channels=channels)
def armroot(case,channels,seed,pilot=False):
    return ROOT/('runs/pilot' if pilot else 'runs/round0')/('fold%d'%case['fold'])/case['key']/('cnn%d-s%d'%(channels,seed))
def selected_model(d,directory,c,device):
    ck=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    m=TransferCNN(1,c['dropout'],c['channels']).to(device);m.load_state_dict(ck['model'])
    return predict(m,d,device,ck['mu'].numpy(),ck['sd'].numpy())
def run_case(d,case,channels,seed,device):
    root=armroot(case,channels,seed)
    if (root/'complete.json').exists():return
    c=config(channels);inner=[]
    for j,split in enumerate(case['inner']):
        r=fit(d,mask(d,split['train']),mask(d,split['validation']),root/('inner%d'%j),c,seed+case['fold']*100+j,'scratch',None,80,12,device,case['test'])
        inner.append(r)
    epochs=int(np.median([r['best_epoch'] for r in inner]))
    score=float(np.average([r['best_validation_MAE'] for r in inner],weights=[len(s['validation']) for s in case['inner']]))
    save(root/'selection.json',dict(config=c,epochs=epochs,inner_MAE=score))
    fit(d,mask(d,case['people']),np.zeros(len(d['ids']),bool),root/'refit',c,seed+case['fold']*100+50,'scratch',None,epochs,12,device,case['test'])
    pp=selected_model(d,root/'refit',c,device)
    save(root/'predictions.json',person_rows(d,pp,case['test']))
    save(root/'complete.json',dict(case=case['key'],fold=case['fold'],channels=channels,seed=seed,inner_fits=3,final_models=1))
def pilot(d,plan,device):
    results=[]
    for case in [c for c in plan['cases'] if c['fold']==0 and ((c['size']==8 and c['draw']==0) or c['size']=='full')]:
        for ch in (8,16,32):
            split=case['inner'][0];out=armroot(case,ch,20261101,True)/'inner0'
            r=fit(d,mask(d,split['train']),mask(d,split['validation']),out,config(ch),20261101,'scratch',None,3,12,device,case['test'])
            pp=selected_model(d,out,config(ch),device)
            save(out/'validation-predictions.json',person_rows(d,pp,split['validation']))
            results.append(dict(path=str(out.relative_to(ROOT)),case=case['key'],channels=ch,**r))
    save(ROOT/'runs/pilot/summary.json',dict(passed=True,cuda=device=='cuda',fits=results,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,torch=str(torch.__version__),python=platform.python_version(),hostname=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID')))
def run_fold(d,plan,fold,device):
    dest=ROOT/('runs/round0/fold%d'%fold)
    if (dest/'complete.json').exists():return
    lock=ROOT/('control/fold%d.lock'%fold);lock.parent.mkdir(exist_ok=True)
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        for case in [c for c in plan['cases'] if c['fold']==fold]:
            for ch in (32,16,8):
                for seed in (20261101,20261102,20261103):run_case(d,case,ch,seed,device)
        save(dest/'complete.json',dict(fold=fold,cases=10,inner_fits=270,final_models=90,total_fits=360))
    finally:lock.unlink()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['check','pilot','train']);ap.add_argument('--round',type=int,default=0);ap.add_argument('--folds',default='0,1,2,3,4');ap.add_argument('--device',default='cuda');a=ap.parse_args()
    assert a.round==0
    for n,h in load(ROOT/'manifest.json').items():assert sha(ROOT/n)==h,n
    d=data(ROOT/'target.npz');plan=load(ROOT/'plan.json');check_plan(plan,d)
    if a.mode=='check':print('{"passed":true,"people":44,"cases":50,"expected_CNN_fits":1800}',flush=True);return
    setup(20261101,a.device)
    if a.mode=='pilot':pilot(d,plan,a.device)
    else:
        for f in map(int,a.folds.split(',')):run_fold(d,plan,f,a.device)
        root=ROOT/'runs/round0'
        if all((root/('fold%d/complete.json'%f)).exists() for f in range(5)):
            try:
                with (root/'summary.lock').open('x') as f:f.write(str(os.getpid()))
            except FileExistsError:return
            save(root/'summary.json',dict(passed=True,outer_folds=5,inner_fits=1350,final_models=450,total_fits=1800))
            save(root/'summary_complete.json',dict(passed=True,summary_sha256=sha(root/'summary.json')))
    print('SAMPLE_SIZE_'+a.mode.upper()+'_FINISHED',flush=True)
if __name__=='__main__':main()
