"""Close source audit with explicit label/task/access boundaries, no model fitting."""
from pathlib import Path
import json,csv,io,re,hashlib
H=Path(__file__).resolve().parent
B=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104')
def main():
    follow=json.loads((H/'visual-function-followup.json').read_text(encoding='utf8'));samples=json.loads((H/'visual-function-samples.json').read_text(encoding='utf8'))
    rows=list(csv.DictReader(io.StringIO((B/'PDFE-PDFEinfo.csv').read_text(encoding='cp1252')),delimiter=';'))
    rd={r['ID'].strip():r for r in rows if re.fullmatch(r'PDFE\d{2}',r['ID'].strip())};inv=json.loads((B/'PDFE-video-inventory.json').read_text());matched=[];excluded=[]
    for x in inv:
        m=re.fullmatch(r'Videos/(PDFE\d{2})_(\d)\.mp4',x['name']);pid,session=m.groups();value=rd[pid][f'Session {session} - TUG time (s)']
        try:v=float(value.replace(',','.'));assert v>0;matched.append({'person':pid,'session':int(session),'tug_seconds':v,'member':x['name']})
        except (ValueError,AssertionError):excluded.append({'person':pid,'session':session,'value':value})
    assert len(rd)==35 and len(inv)==73 and len(matched)==71
    for x in samples['MMU_pose_pilot']:
        p=Path(x['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];z=json.loads(p.read_text());assert z
        if x['mode']=='2D':assert all(len(q['keypoints'])==78 for q in z)
        else:assert all(len(q['keypoints'])==87 and len(q['pred_xyz_jts'])==24 for q in z)
        frames=[int(q['image_id'].split('.')[0]) for q in z];assert len(frames)==len(set(frames)) and frames==list(range(len(frames)))
    mmu={**follow['MMU_FRiP'],'actual_sample_files':4,'sample_people':2,'frame_counts':[256,608],
      'schema_note':'26imagepointsin2D,29imagepointsin3Djsonbut24pred_xyz_jts;neverassumesamejointmapping. JSONscore=detectionconfidence,clinicalscore=separateJH-FRATCSV.',
      'limits':['JH-FRATincludesnonmotoritems,notTUGorSPPB','estimated3D,notmotioncapturetruth','noRGBforownMediaPipeaccuracy','absolutevelocityuncalibrated;onlyrelative2Dfeaturesuntilcalibrationverified'],
      'source':'https://www.kaggle.com/datasets/limzhekhae1008/mmu-frip'}
    pd={'dataset':'PDFEturning','people':35,'age65plus':sum(float(r['Age'].replace(',','.'))>=65 for r in rd.values()),'actual_video_members':73,'positive_TUG_video_sessions':71,
      'TUG_range':[min(x['tug_seconds'] for x in matched),max(x['tug_seconds'] for x in matched)],'matched_sessions':matched,'missing_TUG_sessions':excluded,
      'movement':'2minstanding360degreeturning,Sony30Hzcamera','status':'clinicalCSVdownloadMD5verified;ZIPvideoindexverified;noRGBpayloadorposeinferenceyet',
      'limits':['PDonly','turningtasknotstraightwalking','TUGfromseparateassessment,notvideoduration','noverifiedmetriccalibrationforlinearspeed'],
      'license_note':'FigsharecurrentlicenseCCBY4;paperstatesCC0;retainrepositorylicenseuntilresolved','source':'https://doi.org/10.6084/m9.figshare.14984667'}
    out={'status':'source_and_small_pilot_audit_complete_no_new_accuracy_evaluation',
      'primary_criteria':['ActualRGBorjointtrajectory','IndependentactualTUGorSPPBpreferred','Personandvisitlink','Camerataskcalibrationverifiedbeforeinference','Samepersonfoldandtrainonlyselection','NoIMUfeaturesasprimarymobileaccuracyproof'],
      'existing_ready':[{'dataset':'OlderAdults44','people':44,'age65plus':31,'movement':'opticalmotioncapturejointcentres,220trials','score':'SPPBtotal5..12','status':'locallyanalysedreuseddevelopmentdata,noRGBextractorvalidation','source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC7701341/'},
        {'dataset':'NeuromuscularOpenCap','public_TUG_people':353,'public_TUG_age65plus':45,'already_extracted_eligible_people':297,'already_extracted_age65plus':34,'movement':'smartphonevideo-derived3Dtrajectories,noRGBreleased','score':'actualTUGseconds,samepersonvisitseparateassessment','status':'native3Dfunctionalpredictionpossible;297alreadydevelopmentdata','source':'https://zenodo.org/records/22309771'}],
      'new_public_options':[{'dataset':'MMU-FRiPv2',**mmu},pd],
      'request_only':[{'dataset':'UDDGait','people':304,'movement':'frontalandsideTUGvideo','score':'stopwatchTUGtimes','status':'reasonabledatarequestrequired;noaccessorcontactperformed','source':'https://link.springer.com/article/10.1186/s12877-025-05828-6'},
        {'dataset':'TCAI-FallNet688','people':688,'ages':'65..92','movement':'frontalRGBMediaPipe33joints,SPPBtaskbattery','target':'custom0..100index,40pointsfromvideo-derivedtrunkfeatures;notindependentSPPB0..12gold','status':'rawRGBandkeypointsnonpublic;requestoriginalclinicalSPPB/TUGbeforeprimaryevaluation','source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC13306775/'}],
      'not_primary':[{'dataset':'KINECAL','people':90,'status':'register90rowshasbinaryriskflagbutnonumericTUG/SPPBcolumn;independentnumericlabelsunconfirmed'},
        {'dataset':'SCAI-SCI-Gait','advertised_people':240,'status':'GitHubdocumentationavailablebutlinkedPhysioNetlanding404;credentialedaccessdescribed,notdownloadready;clinicalscorebandsnotTUG/SPPB'}],
      'next_evaluation_design':'KeepTUG/SPPBprimary;directnativejointtrajectorieswithoutsyntheticcamera projection. MMUcanserveaseparatepose-onlyJH-FRATsecondarybenchmark,neverpoolitsscoreswithTUG/SPPB. PDFEcanserveaseparateturningRGBbenchmarkaftercameraQA. Previouslydevelopedcohortsarenotnewexternalconfirmation.',
      'operational_notes':['InitialKINECALparsingusedinvalidutf8-sigalias;correctedtoliteralutf-8-sigandread90rows','Initialpilotshapeassertionincorrectlyexpected26imagepointsinboth2D/3Dfiles;correctjointschemasexplicitlycheckedwithoutdroppingrecords'],
      'no_new_training':True,'no_author_contacts':True,'no_account_or_DUA_acceptance':True}
    (H/'visual-function-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print('VERIFIED MMU125 indexedlabeljoins+4actualposefiles; PDFE35people73videoentries71TUGsessionlinks; KINECAL90rows',flush=True)
if __name__=='__main__':main()
