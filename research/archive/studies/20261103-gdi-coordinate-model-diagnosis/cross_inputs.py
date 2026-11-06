from prepare import *
def main():
 d=np.load(B/'summaries.npz');audit=read(H/'ridge-audit.json');ids=d['person_id'];y=d['y'];arrays={};report={}
 for trained,folds in audit.items():
  report[trained]={}
  for tested in audit:
   out=np.full(y.shape,np.nan)
   for row in folds:
    assert sha(row['checkpoint'])==row['sha256'];m=np.load(row['checkpoint']);te=m['test'];out[te]=(((d[tested][te].reshape(-1,96)-m['mean'])/m['scale'])@m['weight']+m['intercept']).reshape(-1,2)
   assert np.isfinite(out).all();arrays[trained+'__'+tested]=out;report[trained][tested]=float(np.mean([np.abs(out[ids==p]-y[ids==p]).mean() for p in np.unique(ids)]))
 np.savez_compressed(B/'cross-input-predictions.npz',**arrays);save(H/'cross-input-results.json',dict(addendum_sha256=sha(H/'cross-input-addendum.json'),matrix_mae=report,predictions_sha256=sha(B/'cross-input-predictions.npz'),new_fits=0));print(report)
if __name__=='__main__':main()
