from common import *

def main():
    assert not (HERE/'matching-summary.json').exists()
    protocol={'scope':'posthoc descriptive matching, not new held-out performance or causal identification',
        'cohort':'same179people, same visit/session/start/exact6s; sub-283 excluded',
        'features':'3speed summaries plus14original patterns; all17 finite for a feature match; no imputation in pair distance',
        'scale':'per-feature IQR of179 people, outcomes not used; descriptive full-cohort scaling, never used to claim held-out performance',
        'distance':'sqrt(0.5*mean(square(speed difference/IQR))+0.5*mean(square(pattern difference/IQR)))',
        'primary_match':'both pass prior frozen stability rule; mean-speed difference<=0.10m/s; feature distance<=0.5',
        'secondary_match':'same calipers and complete features but without stability exclusion; retain quality flags',
        'large_label_gap':'absolute actual5xSTS difference>=10s; descriptive case-finding threshold, not clinical significance threshold',
        'case_selection':'among qualified pairs order by largest target gap then smallest input distance; up to3 disjoint primary pairs and up to2 disjoint flagged pairs',
        'fallback':'if no primary matches meet gap criterion, report no qualifying pair; do not relax matching thresholds',
        'limitations':'similar current summaries do not establish similar raw motion or identical functional capacity; original RGB unavailable in current files'}
    save('protocol.json',protocol)
    d=load();q=pd.read_csv(QUALITY/'quality-audit.csv');q=q[q.dataset=='STS_common6'].set_index('id').loc[d['ids']]
    assert len(d['ids'])==len(set(d['ids']))==179 and 'sub-283' not in d['ids']
    z=d['pattern'];scale=np.nanpercentile(z,75,axis=0)-np.nanpercentile(z,25,axis=0);scale=np.where(scale>1e-8,scale,1.)
    complete=np.isfinite(z).all(1);rows=[]
    for i in range(len(z)):
        for j in range(i+1,len(z)):
            ds=abs(float(z[i,0]-z[j,0]));finite=bool(complete[i] and complete[j])
            dist=None;robust_dist=None
            if finite:
                v=(z[i]-z[j])/scale;dist=float(np.sqrt(.5*np.mean(v[:3]**2)+.5*np.mean(v[3:]**2)))
            rows.append({'id_a':d['ids'][i],'id_b':d['ids'][j],'i':i,'j':j,'speed_a':float(z[i,0]),'speed_b':float(z[j,0]),
                         'speed_difference':ds,'feature_distance':dist,'complete':finite,'both_stable':bool(q.iloc[i].stable and q.iloc[j].stable),
                         'sts_a':float(d['y'][i]),'sts_b':float(d['y'][j]),'sts_gap':float(abs(d['y'][i]-d['y'][j]))})
    pairs=pd.DataFrame(rows);pairs['matched']=pairs.complete&(pairs.speed_difference<=.1)&(pairs.feature_distance<=.5)
    pairs['primary']=pairs.matched&pairs.both_stable;pairs['discordant']=pairs.sts_gap>=10
    pairs.to_csv(HERE/'all-pairs.csv',index=False)
    selected=[];used=set()
    for group,n in [('primary',3),('flagged',2)]:
        mask=pairs.primary if group=='primary' else pairs.matched&~pairs.both_stable
        candidates=pairs[mask & pairs.discordant].sort_values(['sts_gap','feature_distance'],ascending=[False,True])
        got=0
        for row in candidates.to_dict('records'):
            if row['id_a'] in used or row['id_b'] in used:continue
            row['group']=group;row['case']=f'{group}-{got+1}';selected.append(row);used.update([row['id_a'],row['id_b']]);got+=1
            if got==n:break
    pd.DataFrame(selected).to_csv(HERE/'selected-pairs.csv',index=False)
    # Retain nearest input neighbors for every complete participant, without score selection.
    near=[]
    for pid in d['ids'][complete]:
        a=pairs[pairs.complete & ((pairs.id_a==pid)|(pairs.id_b==pid)) & (pairs.speed_difference<=.1)].sort_values('feature_distance').head(3)
        for rank,r in enumerate(a.to_dict('records')):near.append(dict(anchor=pid,rank=rank+1,**r))
    pd.DataFrame(near).to_csv(HERE/'nearest-input-neighbors.csv',index=False)
    summary={'people':179,'stable_people':int(q.stable.sum()),'complete_feature_people':int(complete.sum()),'all_pairs':len(pairs),
             'matched_pairs':int(pairs.matched.sum()),'matched_large_gap':int((pairs.matched & pairs.discordant).sum()),
             'primary_pairs':int(pairs.primary.sum()),'primary_large_gap':int((pairs.primary & pairs.discordant).sum()),'selected_pairs':len(selected),
             'sources':{str(PRIOR/'STS_common6/input.npz'):sha(PRIOR/'STS_common6/input.npz'),str(QUALITY/'quality-audit.csv'):sha(QUALITY/'quality-audit.csv'),str(LABELS):sha(LABELS)},
             'iqr_scale':scale.tolist(),'protocol_sha256':sha(HERE/'protocol.json')}
    save('matching-summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['sources','iqr_scale']},indent=2));print(pd.DataFrame(selected).to_string(index=False))

if __name__=='__main__':main()
