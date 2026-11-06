"""Build a source audit and standalone scientific plot from verified outputs."""
from pathlib import Path
import json,collections,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import measurement as m

def main():
    v=m.read(m.H/'verification.json');assert v['status']=='passed'
    assert v['results_sha256']==m.sha(m.H/'results.json')
    b=Path('D:/gait-research-data/D_P/planar-measurement-validation-20261105')
    control=pd.read_csv(b/'pd-change/data/study1_control_strides.csv');pddata=pd.read_csv(b/'pd-change/data/study2_patient_gait_metrics.csv')
    paired={field:len(pddata.pivot(index='participant',columns='medication_state',values=field).dropna()) for field in ['gait_speed_world_grounded_m_s','stride_length_world_grounded_m','gait_speed_camera_frame','stride_duration_s','updrs3_gait_item']}
    names=[x['dataFile']['filename'] for x in m.read(Path('D:/gait-research-data/MoVi/control/provider-metadata.json'))['data']['latestVersion']['files']]
    dual=m.read(m.H/'extra-acquisition.json')['dual21_markerless_directory']['entries']
    audit={'date':'2026-11-05','availability_levels':'paper inspected / metadata or archive directory inspected / actual files acquired / actually analyzed are distinct',
      'MoVi':{'source':'https://doi.org/10.5683/SP2/JRHDRN','paper':'https://pmc.ncbi.nlm.nih.gov/articles/PMC8211257/','current_analyzed_people':30,'ages':[18,33],'current_round':'F only','repeat_round_file_counts':{p:sum(n.startswith(p) for n in names) for p in ['S1_PG1','S1_PG2','S2_PG1','S2_PG2']},'repeat_rounds_locally_acquired':False,'limits':'S1/S2 differ in inter-action rest/transitions; sparse optical markers vs F; phone CP footage handheld and not assumed equivalent to synchronized fixed PG cameras; no clinical longitudinal change labels'},
      'GPJATK':{'source':'https://zenodo.org/records/15130800','people':32,'sequences':166,'status':'metadata and release agreement acquired; 6.05GB archive not acquired','reference':'synchronized calibrated video and Vicon','repetition':'walking recordings and clothing/backpack conditions; not rehabilitation longitudinal change','access':'scientific use; release agreement; commercial use/modification requires prior permission; no agreement submitted'},
      'PD_change_tables':{'source':'https://doi.org/10.5281/zenodo.22097474','status':'actual 40,708-byte archive acquired; MD5 and ZIP CRC passed; raw RGB absent; author code read but not executed','healthy_people':int(control.participant.nunique()),'healthy_rows_not_people':len(control),'healthy_source_counts':control.source.value_counts().to_dict(),'PD_people':int(pddata.participant.nunique()),'PD_state_rows':len(pddata),'state_counts':pddata.medication_state.value_counts().to_dict(),'paired_complete_fields':paired,'stride_duration_times_cadence_median':{'control':float((control.stride_duration_s*control.cadence_steps_min).median()),'PD':float((pddata.stride_duration_s*pddata.cadence_steps_min).median())},'unit_flag':'120 vs 60 despite same stride-duration/cadence definitions: unresolved temporal convention, do not silently rescale','limits':'No raw video or per-side steps; camera-frame scale relative and combined speed mixes unit semantics; no independent marker true-change labels for PD; processed clinical condition-change candidate only'},
      'Stenum2024':{'source':'https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000467','stroke_people':44,'PD_people':19,'protocol':'same-day preferred and fast, four walks each; compare camera change with motion-capture change','availability':'clinical raw videos unavailable; public code and four example archives do not equal full cohort'},
      'Older44_GAITRite':{'source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC6983253/','people':44,'ages':[65,91],'walks':264,'protocol':'three preferred + three fast walks with simultaneous GAITRite, Kinect and smartphone','status':'paper inspected; publicly downloadable raw dataset not confirmed; not acquired','limits':'different 44-person study from our previously acquired OlderAdults44 marker/SPPB cohort'},
      'Dual21':{'source':'https://doi.org/10.5281/zenodo.19720803','people':21,'population':'male physical-education students','status':'metadata and markerless ZIP central directory checked, not full archive integrity','entries':len(dual),'csv_files':sum(Path(x['name']).suffix.lower()=='.csv' for x in dual),'video_files_in_inspected_archive':sum(Path(x['name']).suffix.lower() in ['.mp4','.avi','.mov'] for x in dual),'limits':'CSV trajectories only in inspected markerless archive; cannot rerun RGB pose extraction; controlled speed conditions not elderly longitudinal change'},
      'Clemson':{'people':30,'status':'prior metadata/small annotation archive acquired; raw-video extraction not done','limits':'following-camera lower-body footage; temporal unit/sign/origin and reuse conditions unresolved; no independent spatial truth; not counted as current fixed-phone validation'},
      'OpenCap':{'status':'existing video-derived 3D cannot independently validate video measurement by reprojecting it','limits':'not independent optical ground truth'},
      'HealthAndGait398':{'people':398,'ages':[19,64],'status':'provider processed tables; raw RGB unavailable; video and OptoGait separate walking passes','limits':'disagreement includes actual between-pass walking variation; not pure simultaneous measurement error'},
      'BioCV':{'source':'https://researchdata.bath.ac.uk/1258/','people':15,'status':'provider description inspected; request-based access, not acquired','potential':'synchronized video, optical markers and force plates; useful stronger foot-contact reference'}}
    m.save(m.H/'external-data-audit.json',audit)
    pairs=pd.read_csv(m.H/'measurement-pairs.csv');fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for ax,field,label,factor in zip(axes,['speed','step_length','step_time'],['Speed (m/s)','Step length (cm)','Mean step time (s)'],[1,100,1]):
        for camera,color in [('PG1','#126b8b'),('PG2','#c65d32')]:
            d=pairs[(pairs.group=='primary2s')&(pairs.camera==camera)&(pairs.metric==field)]
            ax.scatter(d.reference*factor,d.prediction*factor,label=f'{camera}, n={d.subject.nunique()}',color=color,alpha=.8)
        lower,upper=ax.get_xlim();yl,yu=ax.get_ylim();lo=min(lower,yl);hi=max(upper,yu)
        ax.plot([lo,hi],[lo,hi],color='grey',ls='--',lw=1);ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Optical marker reference',ylabel='Actual RGB / MediaPipe',title=label);ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Fixed primary windows: 14 young adults, 2-second straight walking\nPG1 near-side view; PG2 oblique view. Kinematic contact reference, not force-plate gold.',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(m.H/'measurement-agreement.png',dpi=180);fig.savefig(m.H/'measurement-agreement.pdf');plt.close(fig)
    state={'status':'completed_bounded_measurement_validation_and_source_audit','not_completed_claims':['independent repeated-capture reliability','clinical longitudinal change detection','older adult or real phone accuracy','population asymmetry accuracy'],'verification':v['status'],'primary_people':14,'acquired_people':30,'registered_windows_views':376,'independently_replayed':174,'reference_rejected':202,'event_matching_replayed':True,'reference_and_predicted_step_tables_replayed':True,'one_person_CI_suppressed':True,'next_priority':'Independent floor calibration and longer side-walking sequences; actual repeat captures with independent reference before change thresholds','review_sha256':m.sha(m.H/'review.py'),'source_audit_sha256':m.sha(m.H/'external-data-audit.json')}
    m.save(m.H/'STATE.json',state)
    files=[p for p in m.H.iterdir() if p.is_file() and p.name!='artifact-manifest.json']
    m.save(m.H/'artifact-manifest.json',{'files':{p.name:{'bytes':p.stat().st_size,'sha256':m.sha(p)} for p in sorted(files)},'heavy_root':str(b),'source_hashes_record':'input-audit.json'})
    print(json.dumps({'status':'built','artifact_files':len(files),'source_datasets':len(audit)-2}))

if __name__=='__main__':main()
