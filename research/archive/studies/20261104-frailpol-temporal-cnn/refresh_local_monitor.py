"""Estimate whole unfinished models from completed case timings, never remaining-time halves."""
from pathlib import Path
import json,datetime,argparse
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');G=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');O=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104/native-motion');R=G/'native-motion';CONDS=['phase_tail4','native_tail4','native_tail16']
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
ap=argparse.ArgumentParser();ap.add_argument('--confirmed-live-utc',required=True);args=ap.parse_args();records={};finished=[]
for c in CONDS:
    records[c]=[]
    for f in range(5):
        path=O/c/f'fold{f}/fold.json'
        if path.exists():
            a=read(path);sec=sum(r['elapsed_seconds'] for r in a['inner'])+a['final']['elapsed_seconds'];records[c].append(dict(fold=f,fit_elapsed_seconds=sec,source=str(path),final_epoch=a['epoch']))
    if len(records[c])==5:finished.append(c)
latest_profile=next((c for c in reversed(CONDS) if records[c]),None);totals={};evidence={}
for c in CONDS:
    if c in finished:continue
    source=c if records[c] else latest_profile
    if source is not None:
        totals[c]=5*sum(r['fit_elapsed_seconds'] for r in records[source])/len(records[source]);evidence[c]=dict(observed_cases=len(records[source]),source_condition=source,case_timings=records[source],formula='5*mean(sum3innerfits+selectedfinalfit) fromcompletedcases',limits='Wholemodelplan,notremainingtime. Unstartedconditionborrowslatestmatchingarchitecture/shape/backendprofile;differentselectedepochsandCPUloadremainuncertain.')
    else:raise RuntimeError('No measured case timing; retain pilot estimate instead of inventing one')
now=datetime.datetime.now(datetime.timezone.utc);m=read(R/'local-monitor-state.json');history=m.get('estimate_history',[]);history.append(dict(previous_total_estimates=m.get('total_estimated_seconds_by_unfinished_model'),previous_interval=m.get('interval_seconds'),previous_next=m.get('next_not_before_utc')));m['estimate_history']=history;m['last_check_utc']=now.isoformat();m['completed_models']=finished;m['last_observed_cases']=sum(map(len,records.values()));m['last_observed_case']=read(R/'progress.json')['last'];m['total_estimated_seconds_by_unfinished_model']=totals;m['estimate_evidence']=evidence;m['shortest_unfinished_model_total_estimated_seconds']=min(totals.values()) if totals else None;m['interval_seconds']=min(totals.values())/2 if totals else None;m['next_not_before_utc']=(now+datetime.timedelta(seconds=m['interval_seconds'])).isoformat() if totals else None;m['last_actual_handle_check']='68233and65045bothconfirmedliveat'+args.confirmed_live_utc+';timebasednativeprogressquery';save(R/'local-monitor-state.json',m)
s=read(H/'STATE.json');s['updated_utc']=now.isoformat();s['native_gpu']['last_observed_cases']=m['last_observed_cases'];s['native_gpu']['monitor_next_not_before_utc']=m['next_not_before_utc'];s['native_gpu']['monitor_interval_seconds']=m['interval_seconds'];s['native_gpu']['total_estimated_seconds_by_unfinished_model']=totals;s['native_gpu']['monitor_estimate']=evidence;s['native_gpu']['monitor_note']='Recomputedfromcompletedcasefull3inner+selectedfinalfit times;each5foldmodeltotalprojection,NOTremainingtime. Sourceprofileandhistoryretained. Bothactualtraining/reviewhandlesconfirmedlive18:33UTC. Nextnormalqueryatlistedtimestampunlessfailure/newusersteering.';s['norm_aux']['last_observed_cases']=read(G/'norm-aux/progress.json')['complete'];s['round08']['review_observed_cases_passed']=7;s['next']='Keepallgoalsactive/unchanged. Round08trainingexit0,fullindependentreview36344alive(7/10caseslastobserved);gate74754runsfrozenround10onlyafterpass. NormauxCPU64655/review60158alive,5/25caseslastobserved;all325checkpoint+auxlogitreviewsqueued. NativeGPU68233/review65045alive,last7/15cases;respectlocal-monitor-state nextquerytimestamp/halffullmodelestimate,notremaininghalf. Readfullcomparisons/uncertainty/scorebandsafterreplay. Do notsubmit/change/cancelotherjobs/automations. Allfailedroundsmaintained. Schoolactualclinicaldataexportstillpendingexplicitapproval;localtrainingcontinues. Originalmetadata771butavailableIMU668;only594validfrozenevaluation,9slow. Extra103metadataarenotnewgaitinputs.';save(H/'STATE.json',s);print(json.dumps(dict(completed=finished,estimates=totals,interval_seconds=m['interval_seconds'],next_not_before_utc=m['next_not_before_utc'])))
