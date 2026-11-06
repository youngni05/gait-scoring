"""Post-hoc fixed-order 1/3/5 trial aggregation on saved OOF predictions only."""
from common import *
import pandas as pd

save('aggregation-protocol.json',{'status':'post-hoc descriptive input-count sensitivity; not prespecified primary model selection','fitting':'reuse original OOF models/predictions; no extra fitting','people':44,'trial_order':'source CSV filename ascending; first1/3/5; selection independent of scores/prediction errors','window':'each actual3second walking input','metric':'mean of absolute person error after averaging selected trial predictions','scope':'sensitivity of these44person OOF predictions; not independent confirmation'})
d=load();man=pd.read_csv(HERE/'manifest.csv',dtype={'id':str})
cpu=pd.read_csv(HERE/'cpu-predictions.csv',dtype={'id':str})
gpu=pd.read_csv(HERE/'gpu-return/gpu-predictions.csv').pivot(index='sample',columns='model',values='prediction')
df=man[['sample','id','file','target']].merge(cpu[['sample','Mean','Median','Ridge','RF']],on='sample',validate='one_to_one').merge(gpu,on='sample',validate='one_to_one')
assert len(df)==220 and df.groupby('id').size().eq(5).all()
results=[]
for k in [1,3,5]:
    chosen=df.sort_values('file').groupby('id').head(k)
    assert chosen.groupby('id').size().eq(k).all()
    g=chosen.groupby('id')[['target','Mean','Median','Ridge','RF','CNN','TCN']].mean()
    results.append({'trials':k,'seconds_per_trial':3,'total_selected_seconds':3*k,**{m:float((g[m]-g.target).abs().mean()) for m in ['Mean','Median','Ridge','RF','CNN','TCN']}})
pd.DataFrame(results).to_csv(HERE/'aggregation-sensitivity.csv',index=False)
print(pd.DataFrame(results).to_string(index=False))
