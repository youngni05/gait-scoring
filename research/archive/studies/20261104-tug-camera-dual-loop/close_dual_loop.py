"""Close verified local loops without replacing the original reference model."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
H=Path('D:/Projects/D_P/studies/20261104-tug-camera-dual-loop');B=Path('D:/gait-research-data/D_P/tug-camera-dual-loop-20261104')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def main():
    a=read(H/'verification.json');b=read(H/'followup-verification.json');assert a['status']==b['status']=='passed'
    for rec,filename,held in [(a,'comparison.json','heldout.json'),(b,'followup-comparison.json','followup-heldout.json')]:assert sha(H/filename)==rec['comparison_sha256'] and sha(B/held)==rec['heldout_sha256']
    first=read(H/'comparison.json');follow=read(H/'followup-comparison.json');count=sum(len(s) for s in first.values())+sum(len(s) for s in follow.values());models=len(list(B.rglob('*.joblib')));assert models==9140,models
    # Rebuild local-image-flow aggregation from saved per-frame signals and masks.
    receipt=read(H/'optical-flow-receipt.json');flow=np.load(B/'camera_round1/optical_flow-data.npz');reconstructed=[]
    for r in receipt['records']:
        assert sha(r['path'])==r['sha256'];d=np.load(r['path']);assert np.all(np.diff(d['source_frames'])==1);f=[]
        for j in range(6):
            x=d['signals'][:,j][d['mask'][:,j]];f.extend([float(np.median(x)),float(np.std(x)),float(np.quantile(abs(x),.9))] if len(x)>=3 else [np.nan]*3)
        reconstructed.append(f)
    assert np.array_equal(reconstructed,flow['X'],equal_nan=True)
    state=dict(status='completed_and_independently_verified',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_thread='01a1a1c2-655b-79a3-87bc-e325af86028e',loop_batches=7,all_nonbaseline_policies=129,new_saved_model_artifacts_replayed=models,sensor_raw_base_CNN_retrained=False,local_CPU_only=True,new_school_jobs_or_uploads=0,phone_worn_inputs_used=False,primary_reference='Original0.8805398997851134 retained. New bagmedian0.87409197 is exploratory and pairedCI includeszero; slow9MAE2.98180682 versus original2.89324877.',camera_candidate='Mechanics+fixedtemporalfeatures2.35580735 versus previous2.44284999;14people,pairedCI[-.82792338,.61312266],oneTUG25predicted12.40061104. No establishedgain.',proxy_result='Distillation gives no improvement over historical0.982 camera-compatible-feature development baseline. Best nonqualityproxy1.00936301, speedstudent1.01059574. Availability5 .97521884 remains source-detector diagnostic,notphonehealthinput.',source_modality='594IMUpeople,9TUG>=20;14realvideopeople,1TUG>=20. Sensor-derivedcamera-compatiblequantitiesdistinctfromactualvideoextraction. Actualcamera lacks metric calibration; no matched validcamera-speed-only comparison.',verification={str(H/f):sha(H/f) for f in ['protocol.json','followup-protocol.json','verification.json','followup-verification.json','comparison.json','followup-comparison.json','dual_loop.py','followup_loop.py','review_dual_loop.py','review_followup_loop.py']},flow_aggregation_independently_rebuilt=True,verification_arithmetic_corrections='InitialreviewuseddifferentBLASsummationandquantilefloat64inplaceoffloat32median. Failedcode/logspreserved;matchedoriginalarithmeticwithoutrelaxingpredictiontolerance1e-6abs+relative orgradient1e-8.',limits='Bestobservedconditionsarepost-searchdevelopmentcandidates. CIsnotmultiplicityadjusted;held-outpeoplefromrepeatedlyreusedcohortsarenotfreshclinicalvalidation. SmallmeanMAEgainisnotclinicallymeaningfulchangeevidence. No actualphonefullpipelinebenchmark/SPPBvalidation.',next='No unreported training left running. Furthercameraaccuracyclaimsrequirelargerpairedshortvideo+TUGdata,metriccalibrationandheldoutpeople. Existingoriginalexperimentfiles/modelsunchanged.')
    save(H/'STATE.json',state)
    for name in ['sensor-progress.json','proxy_round1-progress.json','camera_round1-progress.json','camera_round2-progress.json']:
        p=H/name
        if p.exists():r=read(p);r['status']='completed_and_independently_verified';r['closeout']=str(H/'STATE.json');save(p,r)
    print(json.dumps(dict(status=state['status'],loop_batches=7,all_nonbaseline_policies=129,model_artifacts=models,flow_aggregation_replayed=True),indent=2))
if __name__=='__main__':main()
