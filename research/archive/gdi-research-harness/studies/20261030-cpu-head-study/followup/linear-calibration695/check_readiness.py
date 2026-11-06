"""Read-only cross-check of concurrently prepared paper-input data."""
import ast,subprocess,sys,os
from datetime import datetime,timezone,timedelta
import run as r
np,p,m,HERE=r.np,r.p,r.m,r.HERE
folder=p.GPU/'cnn-paper-input695'
audit=m.load_json(folder/'input_audit.json');data=p.load(folder/'bundle/data.npz')
base=p.load(p.PARENT/'development695.npz');idx=data['original_row_index']
assert m.digest(folder/'bundle/data.npz')==audit['data_sha256']
assert len(np.unique(idx))==len(idx)==1061 and len(set(data['person_id']))==692
for key in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(data[key],base[key][idx])
assert data['x'].shape==(1061,2,124,12) and np.isfinite(data['x']).all()
windows=0;no_windows=0
for i in idx:
    proof=m.load_json(folder/f'records/{i:04d}.json')
    assert proof['eligible'] and proof['video_id']==str(base['video_id'][i])
    assert m.digest(folder/f'records/{i:04d}.npz')==proof['artifact_sha256']
    rec=p.load(folder/f'records/{i:04d}.npz')
    np.testing.assert_array_equal(rec['x'],data['x'][np.searchsorted(idx,i)])
    assert len(rec['frame_ids'])==124 and (np.diff(rec['frame_ids'])==1).all()
    windows+=len(proof['eligible_sliding_window_starts']);no_windows+=not bool(proof['eligible_sliding_window_starts'])
assert windows==audit['sliding_windows_available_for_retained_records']==11257
assert no_windows==audit['records_without_sliding_window']==4
exporter=HERE/'export_cnn_train_predictions.py'
ast.parse(exporter.read_text(),feature_version=(3,8))
help_result=subprocess.run([sys.executable,'-B',str(exporter),'--help'],capture_output=True,text=True)
assert help_result.returncode==0 and '--results' in help_result.stdout
# Missing model states must fail before creating output; the received copy intentionally excludes checkpoints.
out=HERE/'missing_checkpoint_probe'
assert not out.exists()
probe=subprocess.run([sys.executable,'-B',str(exporter),'--bundle',str(r.CNN/'bundle'),
    '--results',str(r.CNN/'received/results'),'--out',str(out)],capture_output=True,text=True,
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(sys.path)))
assert probe.returncode!=0 and 'latest.pt' in probe.stderr and not out.exists()
r.js('input_readiness_snapshot.json',dict(observed_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
    source_audit=audit,identity_rows_checked=1061,cached_windows_checked=1061,finite_input_values=int(data['x'].size),
    source_data_sha256=audit['data_sha256'],same_fold_ids=True,original_people=695,usable_people=692,
    sliding_window_starts_count=windows,sliding_windows_materialized_by_this_chat=False,
    paper_inputs_generated_by_this_chat=False,full_preprocessing_independently_reproduced=False,
    exporter_python38_syntax=True,exporter_help_passed=True,exporter_missing_checkpoint_refused=True,
    exporter_actual_checkpoint_inference_tested=False))
print(dict(paper_input_people=692,records=1061,sliding_starts=windows,exporter_missing_checkpoint_refused=True))
