"""CPU/GPU virtual optimization + real-data identity-only checks; no GDI CPU fitting."""
from pathlib import Path
import argparse,copy,json
import numpy as np
import torch
import run as r
import stem

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);r.setup(42,a.device);c=r.config();d=r.data();ext,tr,va=stem.load_external(r.ROOT,c)
    assert len(tr)==14 and len(va)==4
    for f in range(5):
        for j in range(5):
            if f==j:continue
            ti,vi,te=r.split(d,f,j);order=r.sample_epoch(d['x'][ti],d['person_id'][ti],99)
            _,counts=np.unique(d['person_id'][ti][d['x'][ti].owner[order[:,0]]],return_counts=True);assert counts.min()==counts.max()
            altered=d['y'].copy();altered[te]+=12345
            np.testing.assert_array_equal(altered[ti],d['y'][ti]);np.testing.assert_array_equal(altered[vi],d['y'][vi])
    rng=np.random.default_rng(51);toy={'x':[rng.normal(size=(2,n,12)).astype('float32') for n in [25,31,47,81,54,68]],
        'cadence':np.array([90,110,130,150,100,120.]),'walk_id':np.array(['t%d'%i for i in range(6)])}
    t=np.arange(4);v=np.arange(4,6);seed=123;c=dict(c,external_data_sha256='virtual-only')
    full=a.output/'external-full';resume=a.output/'external-resume'
    stem.fit_external(toy,t,v,c,seed,2,full,a.device,r)
    stem.fit_external(toy,t,v,c,seed,2,resume,a.device,r,stop_after=1)
    stem.fit_external(toy,t,v,c,seed,2,resume,a.device,r)
    ck=torch.load(full/'latest.pt',map_location='cpu',weights_only=True);cr=torch.load(resume/'latest.pt',map_location='cpu',weights_only=True)
    for k in ck['model']:assert torch.equal(ck['model'][k],cr['model'][k]),k
    changed_labels=dict(toy,cadence=toy['cadence'].copy());changed_labels['cadence'][v]+=1000
    stem.fit_external(changed_labels,t,v,c,seed,2,a.output/'external-altered-validation',a.device,r)
    altered=torch.load(a.output/'external-altered-validation/latest.pt',map_location='cpu',weights_only=True)
    for k in ck['model']:assert torch.equal(ck['model'][k],altered['model'][k]),'Validation labels altered pretrained weights'
    seeds=[r.config()['seed']+f*100+j for f in range(5) for j in ([k for k in range(5) if k!=f]+[50])]
    assert len(seeds)==len(set(seeds))==25
    # Reuse its selected convolution weights through a temporary root, no real checkpoint overwrite.
    oldroot=r.ROOT;temp=a.output/'transfer-root';target=temp/'runs/pretrain/pilot/seed123';target.mkdir(parents=True)
    (target/'selected.pt').write_bytes((full/'selected.pt').read_bytes())
    c.update(initialization='synth',pretraining_stage='pilot')
    r.setup(seed,a.device);model=r.GdiCNN(c['dropout']).to(a.device);proof=stem.initialize(model,c,seed,temp)
    assert proof['all_other_state_unchanged'] and proof['batchnorm_reset'] and proof['rng_unchanged']
    r.setup(seed,a.device);baseline=r.GdiCNN(c['dropout']).to(a.device)
    for k,value in baseline.state_dict().items():
        if not k.startswith('convs.'):assert torch.equal(value,model.state_dict()[k])
    # Actual original 124-frame topology remains unchanged after convolution transfer.
    x=torch.from_numpy(rng.normal(size=(4,12,124)).astype('float32')).to(a.device)
    assert model(x).shape==(4,2)
    y=torch.from_numpy(rng.uniform(40,110,size=(10,2)).astype('float32')).numpy()
    ws=r.WindowSet(rng.normal(size=(20,2,124,12)).astype('float32'),np.repeat(np.arange(10),2),10)
    ids=np.array(['p%d'%i for i in range(10)])
    results={}
    # The GDI runner's save/resume is exercised only with artificial records.
    for variant in ['scratch','synth']:
        cc=dict(c,initialization=variant,loss_variant='mse',name='virtual_'+variant)
        if variant=='synth':r.ROOT=temp
        ident={'fit_seed':seed,'virtual':True};folder=a.output/variant
        m,_,_,h=r.fit(ws,y,ids,None,None,None,cc,2,folder/'full',ident,a.device)
        r.fit(ws,y,ids,None,None,None,cc,1,folder/'resume',ident,a.device)
        mr,_,_,hr=r.fit(ws,y,ids,None,None,None,cc,2,folder/'resume',ident,a.device)
        for k in m.state_dict():assert torch.equal(m.state_dict()[k],mr.state_dict()[k]),k
        assert h[0]['sample_sha256']==hr[0]['sample_sha256']
        results[variant]={'resume_max_error':0,'changed_tensors':sum(not torch.equal(m.state_dict()[k],baseline.state_dict()[k]) for k in m.state_dict())}
        r.ROOT=oldroot
    assert r.load(a.output/'scratch/full/initialization.json')['initial_conv_sha256']==r.load(a.output/'synth/full/initialization.json')['initial_conv_sha256']
    report={'passed':True,'device':a.device,'real_GDI_optimizer_steps':0,'real_person_inner_splits':20,'external_train_walks':14,'external_validation_walks':4,
        'virtual_external_resume_exact':True,'aux_validation_labels_do_not_change_training_weights':True,'distinct_main_pretraining_seeds':len(seeds),
        'convolution_only_transfer':proof,'GDI_fixed124_topology':True,'virtual_GDI':results,
        'manifest_sha256':r.sha(r.ROOT/'manifest.json'),'external_subject_ids_available':False}
    r.save(a.output/'verification.json',report);print(report,flush=True)
if __name__=='__main__':main()
