from pathlib import Path
import json,hashlib,sys,zipfile
sys.path.insert(0,'D:/Projects/D_P/studies/20261103-frozen-transfer-and-deviation')
from prepare import *
from collections import Counter
# Explain previously unmessaged assertions without altering eligible records or fitted models.
audit=read(HERE/'source-audit.json');details=[]
for row in audit:
 if row.get('reason')!='':continue
 with zipfile.ZipFile(row['archive']) as z:(BIG/'audit-current.c3d').write_bytes(z.read(row['member']))
 c=proj.ezc3d.c3d(str(BIG/'audit-current.c3d'));labels=[n.split(':')[-1].strip() for n in c['parameters']['POINT']['LABELS']['value']];rate=float(c['header']['points']['frame_rate']);axis=[]
 for name in ['LASI','RASI']:
  j=labels.index(name);v=c['data']['points'][:3,j,:].T/1000;v[c['data']['meta_points']['residuals'][0,j,:]<0]=np.nan;axis.append(v)
 t,center=proj.resample(np.mean(axis,axis=0)[:,None,:],rate);center=center[:,0,:]
 details.append({'person':row['person'],'member':row['member'],'reason':'invalid_pelvis_endpoint_prevents_predefined_walkway_alignment' if not np.isfinite(center[[0,-1]]).all() else 'assertion_requires_further_review'})
assert all(d['reason']=='invalid_pelvis_endpoint_prevents_predefined_walkway_alignment' for d in details)
save(HERE/'exclusion-detail.json',details)
new=np.load(BIG/'new-model-oof.npz');err=np.abs(new['age65plus_prediction']-new['age65plus_truth']);baseline=np.abs(new['age65plus_baseline']-new['age65plus_truth']);rng=np.random.default_rng(20261103);ix=rng.integers(0,len(err),(5000,len(err)))
uncertainty={'method':'Fixed person-held-out predictions, person bootstrap; does not include retraining uncertainty','mae_degrees':float(err.mean()),'mae_ci95_degrees':np.quantile(err[ix].mean(1),[.025,.975]).tolist(),'baseline_mae_degrees':float(baseline.mean()),'new_minus_baseline_mae_degrees':float((err-baseline).mean()),'difference_ci95_degrees':np.quantile((err-baseline)[ix].mean(1),[.025,.975]).tolist(),'max_absolute_error_degrees':float(err.max())};save(HERE/'new-model-uncertainty.json',uncertainty)
paper={'url':'https://nmbl.stanford.edu/wp-content/uploads/Kidzinski-Nature-Comm.pdf','location':'Methods page7, Video collection','source_video_fps':29.97,'source_distance_m':[3,4],'camera':'fixed position, rotated around vertical axis to follow patient','source_frame_count':500,'experiment_resample_fps':30,'interpretation':'Near-matched paper-level timing, not proof of every public JSON frame interval. Existing protocol retained30fps scenario; no time stretch or selection based on output.'};save(HERE/'camera-source-evidence.json',paper)
status={'status':'completed_and_independently_verified','skill_copies_updated':True,'skill_validation_passed':True,'frozen_existing_model_inference':{'people':58,'trials':142,'checkpoints':54,'training_steps':0,'original_gdi_accuracy_estimated':False},'new_deviation_model':{'people':134,'healthy':86,'stroke':48,'reference_candidates_age65plus':33,'reference_people_each_outer_training_fold':[26,26,26,27,27],'secondary_allhealthy_reference_candidates':86,'primary_scale':'bilateral angular deviation in degrees, not original GDI','deployment_model':str(BIG/'new-model/age65plus-deployment.npz'),'cpu_only':True},'common_older_stroke_comparison_people':25,'main_result':'Both functional associations attenuated strongly after controlling walking speed; new model superiority unproven. Fixed-model result remains toe-proxy synthetic transfer.', 'artifacts':{name:sha(HERE/name) for name in ['protocol.json','preparation.json','results.json','verification.json','new-model-results.json','new-model-uncertainty.json']},'new_scheduler_created':False,'cancelled_gpu_experiment_restarted':False}
save(HERE/'status.json',status);print(json.dumps({'uncertainty':uncertainty,'explained_exclusions':len(details),'status':status['status']}))

