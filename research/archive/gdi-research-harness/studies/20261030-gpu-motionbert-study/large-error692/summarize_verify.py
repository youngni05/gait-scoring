"""Independent aggregation checks and small, reviewable case report."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent;O=H/'analysis'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
d=read(O/'diagnostics.json');a=read(O/'activations.json')
assert a['diagnostics_sha256']==sha(O/'diagnostics.json')
for p,h in d['source_hashes'].items():assert sha(p)==h
with np.load(S/'mean-difference692/analysis/predictions.npz') as z:blend={k:z[k].copy() for k in z.files}
with np.load(S/'cnn-record-loss692/analysis/window_control/unpacked/results/oof_predictions.npz') as z:control={k:z[k].copy() for k in z.files}
for k in ('person_id','video_id','y','fold'):np.testing.assert_array_equal(blend[k],control[k])
top=np.argsort(-np.abs(control['predictions'][:,0]-control['y']).max(1),kind='stable')[:5]
assert top.tolist()==[c['row'] for c in d['cases']]
windows=list(csv.DictReader((O/'windows.csv').open(encoding='utf-8-sig')));summary=[]
for c in d['cases']:
    i=c['row'];side=int(np.argmax(np.abs(control['predictions'][i,0]-control['y'][i])))
    for key,reference in [('control',control['predictions'][:,0]),('historical',blend['CNN'])]:
        p=c['predictions'][key];raw=np.array(p['window_raw']);corr=np.array(p['window_corrected']);slope,intercept=p['calibration']
        np.testing.assert_allclose(raw*slope+intercept,corr,atol=1e-10,rtol=0)
        np.testing.assert_allclose(raw.mean(0)*slope+intercept,reference[i],atol=1e-4,rtol=0)
        for j in range(2):
            from_csv=[float(w[key+'_raw']) for w in windows if int(w['rank'])==c['rank'] and w['side']==['left','right'][j]]
            np.testing.assert_array_equal(from_csv,raw[:,j])
    summary.append(dict(rank=c['rank'],row=i,side=['left','right'][side],truth=float(control['y'][i,side]),
        control_raw=float(control['predictions_raw'][i,0,side]),control_corrected=float(control['predictions'][i,0,side]),
        historical_CNN=float(blend['CNN'][i,side]),STT=float(blend['STT'][i,side]),
        existing_blend=float(blend['existing_blend'][i,side]),mean_difference=float(blend['inner_selected_difference'][i,side]),windows=c['windows']))
ids=control['person_id'];people=np.unique(ids)
delta=np.array([np.abs(control['predictions'][ids==i,0]-control['y'][ids==i]).mean()-np.abs(blend['CNN'][ids==i]-control['y'][ids==i]).mean() for i in people])
selected_people=ids[top];selected=np.isin(people,selected_people)
report=dict(cases=summary,total_MAE_gap_control_minus_historical=float(delta.mean()),
    five_selected_people_contribution_to_gap=float(delta[selected].sum()/len(people)),
    first_case_person_contribution_to_gap=float(delta[people==ids[top[0]]].sum()/len(people)),
    note='Contribution includes all records of each selected person; no records excluded')
save(O/'case_summary.json',report)
with (O/'case_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
save(O/'verification.json',dict(passed=True,five_cases_selected_reproducibly=True,all_source_hashes_unchanged=True,
    original_raw_sha256_checked=True,raw_to_bundle_exact_match_cases=5,model_case_replays=10,
    cpu_float64_replay_max_error=d['cpu_replay_max_error'],tolerance=1e-4,
    independent_window_aggregation_calibration_and_csv_checked=True,diagnostic_reference_training_only=True,
    diagnostics_sha256=sha(O/'diagnostics.json'),case_summary_sha256=sha(O/'case_summary.json'),
    activation_sha256=sha(O/'activations.json'),protocol_sha256=sha(H/'protocol.json'),
    code_hashes={n:sha(H/n) for n in ('analyze.py','inspect_activations.py','summarize_verify.py')},
    protected_final_opened=False,training_changes=False,optimizer_steps=0))
print(json.dumps(report),flush=True)
