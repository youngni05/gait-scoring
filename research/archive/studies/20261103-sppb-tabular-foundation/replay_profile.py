"""Replay the independent TabICL speed-profile cells while main review checks other cells."""
from foundation import *
import time, gc
def main():
    protocol=read(HERE/'protocol.json')
    for p,h in protocol['sources'].items():assert sha(p)==h
    for f in range(5):
        stem=f'mobilised-speed_profile-{f}-TabICLv2';path=BIG/'predictions'/(stem+'.json');r=read(path)
        dest=BIG/'replay'/(stem+'.json')
        if dest.exists():continue
        assert r['protocol_sha256']==sha(HERE/'protocol.json') and r['factory_sha256']==sha(HERE/'foundation.py')
        assert r['context_sha256']==sha(r['context'])
        with np.load(r['context'],allow_pickle=False) as c:
            assert not set(c['train_ids'])&set(c['test_ids'])
            t=time.monotonic();model=factory('TabICLv2');model.fit(c['X_train'],c['y_train']);p=model.predict(c['X_test'])
        np.testing.assert_allclose(p,r['prediction'],atol=1e-5,rtol=1e-6)
        save(dest,dict(prediction=p.tolist(),seconds=time.monotonic()-t,prediction_sha256=sha(path),context_sha256=sha(r['context']),factory_sha256=sha(HERE/'foundation.py'),worker_code_sha256=sha(Path(__file__))))
        print('replayed',stem,float(np.max(abs(p-np.array(r['prediction'])))),flush=True);del model;gc.collect()
if __name__=='__main__':main()
