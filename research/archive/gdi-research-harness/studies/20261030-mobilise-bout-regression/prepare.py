"""Select paired, same-day bouts without using clinical scores in sampling."""
from common import *
import io,zipfile
from sklearn.model_selection import StratifiedKFold

main=pd.read_csv(SOURCE/'All_cohorts_dataset.csv',usecols=['participantid','visit.number','cohort','total_sppb_score','tugtime']).rename(columns={'visit.number':'visit_number'})
assert not main.duplicated(KEY).any()
main['sppb_ok']=main.total_sppb_score.between(0,12)&main.total_sppb_score.mod(1).eq(0)
main['tug_ok']=np.isfinite(main.tugtime)&main.tugtime.gt(0)
days=pd.read_csv(SOURCE/'bout-person-visit-day-counts.csv')
eligible=days[days.full4_4_10.ge(5)&days.full4_30_60.ge(5)][DAY].merge(main,on=KEY,validate='many_to_one')
eligible=eligible[eligible.sppb_ok|eligible.tug_ok].copy()
chosen=[]
for draw,seed in enumerate(SEEDS):
    e=eligible.copy()
    e['rank']=[stable_int(f'{seed}|{p}|{v}|{day}') for p,v,day in e[DAY].itertuples(index=False,name=None)]
    e=e.sort_values(KEY+['rank']).drop_duplicates(KEY).drop(columns='rank')
    e['draw']=draw; chosen.append(e)
chosen=pd.concat(chosen,ignore_index=True)
chosen.to_csv(HERE/'chosen-days.csv',index=False)
people=chosen[['participantid','cohort']].drop_duplicates().sort_values('participantid').reset_index(drop=True)
assert people.participantid.is_unique
people['fold']=-1
for fold,(_,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEEDS[0]).split(people,people.cohort)):
    people.loc[test,'fold']=fold
people.to_csv(HERE/'person-folds.csv',index=False)
parts=[]; rowcount=0
with zipfile.ZipFile(SOURCE/'mobilise-bouts.zip') as outer:
    for member in outer.namelist():
        if not member.endswith('.zip'):continue
        with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
            with inner.open(inner.namelist()[0]) as stream:
                for d in pd.read_csv(stream,chunksize=200000):
                    rowcount+=len(d)
                    valid=d.datatype.eq('wb')&d.duration_flag.eq(True)&np.isfinite(d.duration)
                    valid &= np.isfinite(d[FEATURES]).all(axis=1)&d[FEATURES].gt(0).all(axis=1)&d[[c+'_flag' for c in FEATURES]].eq(True).all(axis=1)
                    short=d.duration.ge(4)&d.duration.lt(10); long=d.duration.ge(30)&d.duration.lt(60)
                    d=d[valid&(short|long)].copy()
                    d['band']=np.where(d.duration.lt(10),'short','long')
                    d=d[DAY+['band','wb_id','duration','starttimestamputc','turn_number_so']+FEATURES]
                    d=d.merge(chosen[DAY+['draw']],on=DAY,how='inner',validate='many_to_many')
                    if d.empty:continue
                    # Fixed pseudorandom priorities; independent of target values and file chunks.
                    h=pd.util.hash_pandas_object(d[KEY+['wb_id']],index=False).to_numpy(dtype='uint64')
                    x=h+np.array([SEEDS[i] for i in d.draw],dtype='uint64')
                    x=(x^(x>>30))*np.uint64(0xbf58476d1ce4e5b9)
                    x=(x^(x>>27))*np.uint64(0x94d049bb133111eb)
                    d['priority']=x^(x>>31)
                    group=KEY+['draw','band']
                    parts.append(d.sort_values(group+['priority','wb_id']).groupby(group,sort=False).head(5))
        print('Processed',member,flush=True)
selected=pd.concat(parts,ignore_index=True).sort_values(KEY+['draw','band','priority','wb_id']).groupby(KEY+['draw','band'],sort=False).head(5)
assert selected.groupby(KEY+['draw','band']).size().eq(5).all()
selected['rank']=selected.groupby(KEY+['draw','band']).cumcount()+1
assert not selected.duplicated(KEY+['draw','band','wb_id']).any()
selected.to_csv(HERE/'selected-bouts.csv',index=False)
features=[]
for k in [1,3,5]:
    d=selected[selected['rank'].le(k)]
    grouped=d.groupby(KEY+['draw','band'])
    a=grouped[FEATURES].median()
    a['total_seconds']=grouped.duration.sum()
    a['span_seconds']=grouped.apply(lambda x:float((x.starttimestamputc+x.duration).max()-x.starttimestamputc.min()),include_groups=False)
    a=a.reset_index(); a['k']=k; features.append(a)
features=pd.concat(features,ignore_index=True).merge(main,on=KEY,validate='many_to_one').merge(people[['participantid','fold']],on='participantid',validate='many_to_one')
assert np.isfinite(features[FEATURES]).all().all()
features.to_csv(HERE/'features.csv',index=False)
counts={}
for target in ['sppb','tug']:
    d=features[features[target+'_ok']&features.draw.eq(0)&features.k.eq(1)&features.band.eq('short')]
    counts[target]={'people':int(d.participantid.nunique()),'visits':len(d),'fold_people':{str(k):int(v) for k,v in d.groupby('fold').participantid.nunique().items()}}
assert counts['sppb']['people']==2174 and counts['tug']['people']==1939
save('preparation.json',{'counts':counts,'rows_scanned':rowcount,'features':FEATURES,'seeds':SEEDS,'source_sha256':{n:sha(SOURCE/n) for n in ['mobilise-bouts.zip','All_cohorts_dataset.csv','bout-person-visit-day-counts.csv']},'selected_bouts_sha256':sha(HERE/'selected-bouts.csv'),'features_sha256':sha(HERE/'features.csv')})
print(json.dumps(counts,indent=2),flush=True)
