import numpy as np,json
B='D:/gait-research-data/D_P/final222-normal100-20261106/fusion-predictions.npz'
z=np.load(B,allow_pickle=False); ids=z['person_id'].astype(str); y=z['y']
def pers(v,m):
    p=np.unique(ids[m.any(1)]); return np.mean([v[(ids==q)[:,None]&m].mean() for q in p]),len(p)
for arm in ('baseline','mixed_stanford_cal'):
    h=z[arm]; e=h-y
    Y=y.ravel();H=h.ravel()
    print(arm,'r',round(np.corrcoef(Y,H)[0,1],3),'slope pred on true',round(np.polyfit(Y,H,1)[0],3),'slope true on pred',round(np.polyfit(H,Y,1)[0],3),'sd true',round(Y.std(),2),'sd pred',round(H.std(),2),'mean true',round(Y.mean(),2),'mean pred',round(H.mean(),2))
    for lo,hi in [(0,60),(60,70),(70,80),(80,90),(90,200)]:
        m=(h>=lo)&(h<hi)
        if m.sum()==0: print(' pred band',lo,hi,'none');continue
        b,n=pers(e,m); print(' pred band',lo,hi,'obs',int(m.sum()),'people',n,'bias(pred-true)',round(b,3))
    for lo,hi in [(0,50),(50,70),(70,90),(90,100),(100,200)]:
        m=(y>=lo)&(y<hi); b,n=pers(e,m); print(' true band',lo,hi,'obs',int(m.sum()),'people',n,'bias',round(b,3))
# expected regression-to-mean from correlation alone: E[pred-true|true]
Y=y.ravel();H=z['baseline'].ravel();r=np.corrcoef(Y,H)[0,1]
for t in (45,105):
    print('RTM expected bias at true',t,'with slope',round(np.polyfit(Y,H,1)[0],3),round((Y.mean()+np.polyfit(Y,H,1)[0]*(t-Y.mean()))-t+ (H.mean()-Y.mean()),2))
# self video separation
import csv
b={int(r['number']):float(r['score']) for r in csv.DictReader(open('D:/Projects/D_P/gdi-research-harness/studies/20261101-self-video-test01/video-scores.csv',encoding='utf-8-sig'))}
R=json.load(open('D:/Projects/D_P/studies/20261105-unseen-normal-test/stanford-calibration/results.json',encoding='utf8'))
for nm,v in (('before',b),('mixed',{int(k):x for k,x in R['external']['mixed']['videos'].items()}),('stan',{int(k):x for k,x in R['external']['mixed_stanford_cal']['videos'].items()})):
    n=np.array([v[i] for i in (1,2,3,6,8,9)]);a=np.array([v[i] for i in (4,5,7,10,11,12,13)])
    sp=np.sqrt((5*n.var(ddof=1)+6*a.var(ddof=1))/11)
    print(nm,'gap',round(n.mean()-a.mean(),2),'sdN',round(n.std(ddof=1),2),'sdA',round(a.std(ddof=1),2),'d',round((n.mean()-a.mean())/sp,2),'minN-maxA',round(n.min()-a.max(),2),'n',np.round(n,1),'a',np.round(a,1))
