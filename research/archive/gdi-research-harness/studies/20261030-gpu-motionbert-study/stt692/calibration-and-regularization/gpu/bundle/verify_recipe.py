"""Unequal person/record/window counts and actual-fold weight conservation."""
from pathlib import Path
import argparse,numpy as np,torch
import run
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=run.load(run.ROOT/'protocol.json');run.setup(21,'cpu')
    owner=np.array([0,1,1,1,2,2]);ids=np.array(['a','a','b'])
    x=run.WindowSet(np.zeros((6,2,124,4,2),np.float32),owner,3)
    targets=np.arange(12,dtype=np.float32).reshape(6,2)
    w=run.training_weights(x,ids,run.configure(c,'person')).reshape(6,2)
    actual=np.mean(w*targets**2)
    record_errors=np.array([(targets[owner==i]**2).mean() for i in range(3)])
    expected=np.mean([record_errors[ids==q].mean() for q in np.unique(ids)])
    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-7)
    for variant in ('baseline','decay'):np.testing.assert_array_equal(run.training_weights(x,ids,run.configure(c,variant)),x.side_weights())
    # All recipe arms start from exactly the same parameters and predictions.
    initial=[]
    for variant in run.VARIANTS:
        run.setup(23,'cpu');m=run.new_model();opt=run.make_optimizer(m,run.configure(c,variant))
        initial.append({k:v.clone() for k,v in m.state_dict().items()})
        assert opt.param_groups[0]['weight_decay']==(0.01 if 'decay' in variant else 0.)
    assert all(torch.equal(initial[0][k],s[k]) for s in initial for k in initial[0])
    d=run.data();checks=[]
    for f in range(5):
        for j in [None]+[i for i in range(5) if i!=f]:
            tr,va,te=run.split(d,f,j);x=d['x'][tr];ids=d['person_id'][tr]
            w=run.training_weights(x,ids,run.configure(c,'person')).reshape(-1,2)
            np.testing.assert_allclose(w.mean(),1,rtol=0,atol=2e-7)
            totals=np.array([w[ids[x.owner]==q].sum(dtype=float) for q in np.unique(ids)])
            np.testing.assert_allclose(totals,2*len(x.values)/len(totals),rtol=2e-7,atol=2e-7)
            checks.append({'outer':f,'inner':j,'people':len(totals),'max_ratio':float(totals.max()/totals.min())})
    run.save(a.output/'verification.json',{'passed':True,'all25_training_weights_verified':True,'synthetic_hierarchical_loss_verified':True,
        'same_initial_weights':True,'optimizer_decay_settings_verified':True,'real_GDI_optimizer_steps':0,'checks':checks})
    print('RECIPE_VERIFICATION_PASSED')
if __name__=='__main__':main()
