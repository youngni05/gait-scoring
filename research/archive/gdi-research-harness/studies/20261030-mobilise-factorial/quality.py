"""Audit released labels and restore raw turn-quality flags for the exact selected bouts."""
import importlib.util,io,json,zipfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'20261030-mobilise-deep-sets-stability'
spec=importlib.util.spec_from_file_location('reference_study',REFERENCE/'train.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
ref=previous.ref;BASE=previous.BASE;SOURCE=HERE.parents[2]/'outputs/sppb-tug-candidates-20261030'
import pandas as pd
import numpy as np

def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')

def run():
    started=time.time();selected=pd.read_csv(BASE/'selected-bouts.csv');key=['participantid','visit_number','wb_id']
    unique=selected.drop_duplicates(key);lookup=unique[key]
    parts=[];file_checks=[];total=0
    cols=key+['wbday','duration','duration_flag','turn_number_so','turn_number_so_flag','starttimestamputc','visit_date']
    with zipfile.ZipFile(SOURCE/'mobilise-bouts.zip') as outer:
        for name in outer.namelist():
            if not name.endswith('.zip'):continue
            count=0
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                with inner.open(inner.namelist()[0]) as stream:
                    for chunk in pd.read_csv(stream,usecols=cols,chunksize=400000):
                        count+=len(chunk);matched=chunk.merge(lookup,on=key,how='inner',validate='one_to_one')
                        if len(matched):parts.append(matched)
            file_checks.append({'member':name,'rows':count,'crc_read_to_end':True});total+=count
            print('Audited',name,count,'raw rows',flush=True)
    raw=pd.concat(parts,ignore_index=True);assert not raw.duplicated(key).any() and len(raw)==len(unique)
    merged=selected.merge(raw,on=key,validate='many_to_one',suffixes=('','_raw'))
    for name in ['duration','turn_number_so','wbday','starttimestamputc']:
        np.testing.assert_allclose(merged[name],merged[name+'_raw'],atol=1e-10,rtol=1e-12,equal_nan=True)
    valid=raw.turn_number_so_flag.eq(True)&np.isfinite(raw.turn_number_so)&raw.turn_number_so.ge(0)&raw.turn_number_so.mod(1).eq(0)&raw.duration_flag.eq(True)&np.isfinite(raw.duration)&raw.duration.gt(0)
    raw['turn_valid']=valid;raw['turn_rate_per_second']=np.where(valid,raw.turn_number_so/raw.duration,np.nan)
    raw.to_csv(HERE/'audited-bouts.csv',index=False)
    f=pd.read_csv(BASE/'features.csv');f=f[f.draw.eq(0)&f.band.eq('long')&f.k.eq(5)]
    cols=['participantid','visit.number','cohort','visit.date','tugtime','tugaids','tuginst','total_sppb_score','balance_score','4mwt_speed_score','5crt_score','4mwt1secs','4mwt2secs','4mwt1fail','4mwt2fail','5crtresult']
    main=pd.read_csv(SOURCE/'All_cohorts_dataset.csv',usecols=cols,low_memory=False).rename(columns={'visit.number':'visit_number'})
    assert not main.duplicated(['participantid','visit_number']).any()
    selected_labels=f[['participantid','visit_number','sppb_ok','tug_ok','total_sppb_score','tugtime']].merge(main,on=['participantid','visit_number'],validate='one_to_one',suffixes=('_used',''))
    for label in ['total_sppb_score','tugtime']:np.testing.assert_allclose(selected_labels[label],selected_labels[label+'_used'],atol=0,rtol=0,equal_nan=True)
    sppb=selected_labels[selected_labels.sppb_ok].copy();components=['balance_score','4mwt_speed_score','5crt_score'];complete=sppb[components].notna().all(axis=1)
    mismatch=complete&sppb[components].sum(axis=1).ne(sppb.total_sppb_score)
    tug=selected_labels[selected_labels.tug_ok].copy();tug['review_flag']=tug.tugtime.lt(3)|tug.tugtime.gt(60)
    # Concordance with separately released per-cohort tables is a transcription check,
    # not proof that the original stopwatch observation is clinically correct.
    concordance=[]
    with zipfile.ZipFile(SOURCE/'mobilise-main.zip') as z:
        for cohort in ['COPD','MS','PD','PFF']:
            path=next(n for n in z.namelist() if n.endswith('/'+cohort+'_dataset.csv'))
            with z.open(path) as stream:d=pd.read_csv(stream,usecols=['participantid','visit.number','tugtime','total_sppb_score'],low_memory=False).rename(columns={'visit.number':'visit_number'})
            check=selected_labels[selected_labels.cohort.eq(cohort)].merge(d,on=['participantid','visit_number'],validate='one_to_one',suffixes=('','_cohort'))
            assert len(check)==int(selected_labels.cohort.eq(cohort).sum())
            for label in ['tugtime','total_sppb_score']:np.testing.assert_allclose(check[label],check[label+'_cohort'],atol=0,rtol=0,equal_nan=True)
            concordance.append({'cohort':cohort,'rows':len(check),'labels_exact':True})
    audit=selected_labels[['participantid','visit_number','sppb_ok','tug_ok']].copy()
    audit['sppb_component_review']=selected_labels[components].isna().any(axis=1)|selected_labels[components].sum(axis=1).ne(selected_labels.total_sppb_score)
    audit['tug_range_review']=selected_labels.tugtime.lt(3)|selected_labels.tugtime.gt(60)
    audit.to_csv(HERE/'label-review-flags.csv',index=False)
    save('quality.json',{'raw_rows_scanned':total,'files':file_checks,'selected_unique_bouts':len(unique),'selected_bout_rows_all_draws':len(selected),'matched_unique_bouts':len(raw),'turn_invalid_unique_bouts':int((~valid).sum()),'turn_flag_counts':{str(k):int(v) for k,v in raw.turn_number_so_flag.value_counts(dropna=False).items()},'negative_or_fractional_turns':int((raw.turn_number_so.lt(0)|raw.turn_number_so.mod(1).ne(0)).sum()),'duration_flags_valid':bool(raw.duration_flag.eq(True).all()),'turn_max':float(raw.turn_number_so.max()),'turn_rate_max':float(raw.turn_rate_per_second.max()),'values_match_previously_selected_bouts':True,'sppb_people':sppb.participantid.nunique(),'sppb_visits':len(sppb),'sppb_complete_components':int(complete.sum()),'sppb_component_missing':int((~complete).sum()),'sppb_component_sum_mismatch':int(mismatch.sum()),'tug_people':tug.participantid.nunique(),'tug_visits':len(tug),'tug_below3':int(tug.tugtime.lt(3).sum()),'tug_above60':int(tug.tugtime.gt(60).sum()),'tug_review_people':int(tug[tug.review_flag].participantid.nunique()),'tug_range':[float(tug.tugtime.min()),float(tug.tugtime.max())],'cohort_source_concordance':concordance,'label_policy':'retain every original label/person/bout in primary fitting and evaluation; review flags are never predictors. Additionally evaluate predictions on unflagged visits as a sensitivity analysis, without refitting or replacing primary results. No confirmed corrections available.','turn_quality_limit':'released algorithm quality flags and numeric validity, not external video/reference validation of turn detection','elapsed_seconds':time.time()-started,'source_sha256':{str(p):ref.sha(p) for p in [BASE/'selected-bouts.csv',BASE/'features.csv',SOURCE/'mobilise-bouts.zip',SOURCE/'mobilise-main.zip',SOURCE/'All_cohorts_dataset.csv',SOURCE/'mobilise-dictionary.zip']},'artifact_sha256':{n:ref.sha(HERE/n) for n in ['audited-bouts.csv','label-review-flags.csv']}})
    print(json.dumps(json.loads((HERE/'quality.json').read_text()),indent=2),flush=True)
if __name__=='__main__':run()
