from foundation import *
import time, argparse, gc
def main():
    parser=argparse.ArgumentParser();parser.add_argument('cohort');args=parser.parse_args()
    pp=read(BIG/'cohorts.json')[args.cohort];X,names=matrices(args.cohort,pp)['features'];y=np.array([p['sppb'] for p in pp]);f=np.array([p['fold'] for p in pp]);tr=f!=0;te=f==0
    records=[]
    for kind in ('TabICLv2','TabPFNv2'):
        start=time.monotonic();model=factory(kind);model.fit(X[tr],y[tr]);p=model.predict(X[te]);elapsed=time.monotonic()-start
        assert np.isfinite(p).all()
        rec=dict(cohort=args.cohort,model=kind,train_n=int(tr.sum()),test_n=int(te.sum()),features=len(names),seconds=elapsed,prediction_finite=True,configuration=repr(model))
        records.append(rec);save(HERE/f'pilot-{args.cohort}.json',records);print(json.dumps(rec),flush=True)
        del model;gc.collect()
if __name__=='__main__':main()
