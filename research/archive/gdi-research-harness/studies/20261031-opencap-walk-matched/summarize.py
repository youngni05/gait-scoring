from common import *

def main():
    d=load();a=pd.read_csv(HERE/'all-pairs.csv');sel=pd.read_csv(HERE/'selected-pairs.csv')
    scale=np.array(json.loads((HERE/'matching-summary.json').read_text())['iqr_scale'])
    names=['speed_mean','speed_sd','speed_adjacent_change']+quality.PATTERN_NAMES
    rows=[]
    for r in sel.itertuples():
        for k,name in enumerate(names):rows.append(dict(case=r.case,feature=name,a=d['pattern'][r.i,k],b=d['pattern'][r.j,k],difference_in_iqr=abs(d['pattern'][r.i,k]-d['pattern'][r.j,k])/scale[k]))
    pd.DataFrame(rows).sort_values(['case','difference_in_iqr'],ascending=[True,False]).to_csv(HERE/'pair-existing-differences.csv',index=False)
    man=pd.read_csv(PRIOR/'STS_common6/manifest.csv').set_index('id');labels=pd.read_csv(LABELS).set_index(['subid','visit'])
    q=pd.read_csv(QUALITY/'quality-audit.csv');q=q[q.dataset=='STS_common6'].set_index('id').loc[d['ids']]
    f=pd.read_csv(HERE/'extra-features-all.csv').set_index('id').loc[d['ids']]
    zero=(f.knee_r_zero_fraction>=.5)|(f.knee_l_zero_fraction>=.5)
    primary=a[a.primary&a.discordant].copy()
    for suffix in ['a','b']:
        for field in ['age','diag']:
            primary[field+'_'+suffix]=[labels.loc[(pid,int(man.loc[pid].visit)),field] for pid in primary['id_'+suffix]]
    primary['max_existing_difference_iqr']=[float(np.max(abs(d['pattern'][r.i]-d['pattern'][r.j])/scale)) for r in primary.itertuples()]
    primary.to_csv(HERE/'all-primary-discordant-pairs.csv',index=False)
    chairs=json.loads((HERE/'chair-audit.json').read_text());people=pd.read_csv(HERE/'person-audit.csv')
    save('interpretation-checks.json',{'zero_at_least_half_one_knee_people':int(zero.sum()),'zero_at_least_half_stable_people':int((zero.to_numpy()&q.stable.to_numpy()).sum()),
         'primary_discordant_adult_same_diagnosis':primary[(primary.age_a>=18)&(primary.age_b>=18)&(primary.diag_a==primary.diag_b)][['id_a','id_b']].to_dict('records'),
         'selected_all_marker_jump_fractions_zero':bool((people.marker_jump_fraction==0).all()),
         'selected_velocity_disagreement_p95_range':[float(people.velocity_disagreement_p95.min()),float(people.velocity_disagreement_p95.max())],
         'chair_record_shorter_than_label':[{k:r[k] for k in ['id','label','record_seconds','task']} for r in chairs if r['label']>r['record_seconds']],
         'visual_review':{'sub-268':'4 clear complete height excursions plus final rising segment cut by recording end; filename3xsts does not equal observed excursion count. Do not relabel.',
                          'sub-016':'4 complete excursions plus final rising segment; simple peak count cannot verify5th completion.',
                          'sub-071':'5 excursions visible but complete file shorter than clinical label.',
                          'sub-182':'5 excursions visible but complete file shorter than clinical label.',
                          'sub-138_and_sub-304':'5 excursions visible in each, slower cycles for304; exact stopwatch event boundaries still unavailable.'},
         'geometric_knee_limit':'3D hip-knee-ankle included angle differs in definition from model flexion; ordering reversal identifies definition/model sensitivity, not which estimate is true.',
         'same6s_scope':'All new gait quantities use the frozen181 samples or marker data inside the same start/end. Original OpenCap IK may have used the full recording; not prospective6s raw-RGB processing.'})
    print(primary[['id_a','id_b','max_existing_difference_iqr']].to_string(index=False))
if __name__=='__main__':main()
