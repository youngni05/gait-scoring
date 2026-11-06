"""Can an RGB-only orientation-stability gate reject turns without losing straight walks?"""
from pathlib import Path
import importlib.util,json,argparse,time
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paired_cadence_fixed',HERE.parent/'20261101-cadence-straight-control/study.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
np,pd,sha,save=source.np,source.pd,source.sha,source.save
STORE=Path('D:/gait-research-data/D_P/camera-straight-gate-20261101')
def gate(world,image):
    # Operational camera-frame x/z vector stability, not anatomical heading truth.
    good=np.isfinite(world[:,[11,12,23,24],:3]).all() and np.isfinite(image[:,[11,12,23,24],:]).all() and (image[:,[11,12,23,24],3]>=.5).all()
    spans=[]
    for left,right in [(11,12),(23,24)]:
        v=world[:,right,[0,2]]-world[:,left,[0,2]];norm=np.linalg.norm(v,axis=1)
        if not np.isfinite(v).all() or (norm<=1e-6).any():return dict(accepted=False,span=None,landmarks_valid=False)
        angles=np.unwrap(np.arctan2(v[:,1],v[:,0]));spans.append(float(np.rad2deg(np.percentile(angles,95)-np.percentile(angles,5))))
    span=max(spans);return dict(accepted=bool(good and span<=30),span=span,landmarks_valid=bool(good))
def prepare():
    sources=[source.HERE/n for n in ['study.py','protocol.json','main-verification.json','main-estimates.csv','main-events.csv','geometry-paired.csv']]+[source.CLIPS]
    save(HERE/'protocol.json',dict(code_sha256=sha(__file__),sources={str(p):sha(p) for p in sources},
      hypothesis='Cadence-proxy disagreement increases on preselected turns. A fixed RGB-only torso-vector orientation-stability gate may reject turns while preserving straight clips,without using optical geometry as an input.',
      inputs='Same14people x2secstraight/turn xPG1/PG2=56clips;existingMediaPipeimage/local3D. Gateonly shoulder11/12 andhip23/24 local3D x/z axis vectors;imagevisibilitycolumn3. Opticalsourcegeometry is evaluator label only. No video refitting/inference or thresholdsearch.',
      algorithm='All4joint xyzfinite/imagefinite andvisibility>=.5 forall60frames;both left-right x/z vectornorm>1e-6. Unwrapatan2(z,x) pervector;maxshoulder/hip p95-p05span<=30deg.30deg transferred from prior physicalgeometry diagnostic,not chosen after RGBgate outcomes. Operational axis stability,not validated anatomical yaw. No cadence/score outcome entersgate.',
      evaluation='Retain all14people and56clips. Report straightretention andturnfalseacceptance perview,pairedbothviewrates anddisagreement. Report oldperiodquality/intersection counts separately;gated spectral-vs-event discrepancies only where eventquality andfiniteperiod. All denominators,do not call accepted-only accuracy an overall gain.',
      success='Exploratory capturequality engineering target:straightretention>=80% andturnrejection>=80% inEACHview,reportcomponents/joint. No clinical standard. No tuning iffails.',
      limits='Same14developmentpeople previously examined,not independentvalidation. Straight/turnlabels selected by referencebodygeometry,not clinicalscore/contacttruth. Local3Dorientation is RGBprediction,depthnoisecanrejectstraightwalks. Gatecannotprove truecadence,phonegeneralization,SPPB/TUGaccuracy or classifylowfunction. Reject=measurementunavailable,never low score. Originalscorepeople/bouts/5seeds unchanged.',
      expected=dict(main_tasks=14,pilot_tasks=2,people=14,clips=56,gate_replays=56,metric_rows=2,paired_rows=2,proxy_rows=8,new_model_fits=0),
      verification='Frozenarray/source/codeSHA,frame/person/view/condition links,independentmanualphaseunwrap/quantiles/norm/visibility andgate,1e-10span/exactboolean;independent counts/discrepancy recomputation. Pilot2people8clips thenmain.',
      resources='CPU1,noGPU/download/newinference/training,estimate1-3min;selectedarraysD<5MB;code/reportsC',storage=dict(small=str(HERE),large=str(STORE))))
def work(sid):
    path=STORE/f's{sid:02d}.json'
    if path.exists():
        q=json.loads(path.read_text());assert q['code_sha256']==sha(__file__);return q
    parent=source.STORE/f's{sid:02d}.json';q=json.loads(parent.read_text());rows=[];arrays={};files={str(parent):sha(parent)}
    for segment in q['segments']:
        assert sha(segment['array_path'])==segment['array_sha256'];files[segment['array_path']]=segment['array_sha256'];z=np.load(segment['array_path'])
        for cam in ['PG1','PG2']:
            tag=segment['condition']+'_'+cam;world=z[cam+'_world'];image=z[cam+'_lm'];arrays[tag+'_world']=world;arrays[tag+'_image']=image;arrays[tag+'_frames']=z[cam+'_frames'];rows.append(dict(subject=sid,condition=segment['condition'],camera=cam,**gate(world,image)))
    array=STORE/f's{sid:02d}.npz';np.savez_compressed(array,**arrays);out=dict(subject=sid,rows=rows,array_path=str(array),array_sha256=sha(array),sources=files,code_sha256=sha(__file__));save(path,out);return out
def run(mode):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for p,h in protocol['sources'].items():assert sha(p)==h
    STORE.mkdir(parents=True,exist_ok=True);rows=[];started=time.time()
    for i,sid in enumerate(source.ids(mode),1):rows+=work(sid)['rows'];print('GATE',i,sid,flush=True)
    pd.DataFrame(rows).to_csv(HERE/f'{mode}-gates.csv',index=False);save(HERE/f'{mode}-completion.json',dict(tasks=len(source.ids(mode)),people=len(source.ids(mode)),clips=len(rows),new_model_fits=0,seconds=time.time()-started,code_sha256=sha(__file__)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
