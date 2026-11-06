"""Run the already specified CPU comparison, then require its independent audit."""
import datetime,json,time,traceback
import numpy as np,torch
from threadpoolctl import threadpool_limits
import compare
from features import OUT,GdiCNN,channels,sha
import verify_results

def status(phase,**extra):
    compare.save(OUT/'status.json',{'phase':phase,'job':'33443','pose_complete':True,'people':10,
        'updated_at':datetime.datetime.now().astimezone().isoformat(),'consumer_session':None,**extra})

def pilot():
    d=compare.load();fold,tr,te,inner=compare.splits(d)[0];a,b=inner[0]
    assert not set(d['ids'][a])&set(d['ids'][te])
    torch.manual_seed(20261031);initial=GdiCNN(.5).state_dict()
    start=time.time();path=OUT/'training-pilot'
    compare.fit_nn(d,a,None,3,20261031,path)
    ck=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    changed=sum(not torch.equal(v,ck['state'][k]) for k,v in initial.items() if k.endswith('weight'))
    assert changed>0
    predictions=[]
    x=(torch.tensor(d['x'][b])-ck['mu'])/ck['sd']
    for _ in range(2):
        m=GdiCNN(.5).eval();m.load_state_dict(ck['state'])
        with torch.inference_mode():predictions.append(m(channels(x).reshape(-1,12,124))[:,0].reshape(-1,2).mean(1).numpy()*ck['ystd']+ck['ymean'])
    assert np.isfinite(predictions[0]).all();np.testing.assert_allclose(predictions[0],predictions[1],rtol=0,atol=1e-5)
    compare.save(OUT/'training-pilot.json',{'passed':True,'epochs':3,'changed_weight_tensors':changed,'seconds':time.time()-start,
        'train_people':d['ids'][a].tolist(),'replay_people':d['ids'][b].tolist(),'outer_test_people_used':False,'input_sha256':sha(OUT/'prepared.npz')})

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    try:
        with threadpool_limits(limits=2):
            status('CPU actual-data training pilot');pilot()
            status('CPU scalar A/B/C/random-control comparison');compare.scalar()
            status('CPU scratch CNN D comparison');compare.scratch()
            status('Independent final audit');verify_results.main()
        status('comparison_complete_verified',result_sha256=sha(OUT/'comparison.json'),verification_sha256=sha(OUT/'verification.json'))
        print('GDI_TUG_COMPARISON_VERIFIED',flush=True)
    except Exception:
        status('comparison_failed_needs_review',error=traceback.format_exc());raise
if __name__=='__main__':main()
