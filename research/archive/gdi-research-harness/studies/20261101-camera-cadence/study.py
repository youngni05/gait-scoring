"""RGB cadence-period proxy and paired-view observability; no clinical score labels."""
from pathlib import Path
import sys,json,argparse,hashlib,time,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
SOURCE=Path('D:/gait-research-data/MoVi/experiments/walking-expansion')
AUDIT=HERE.parent/'20261031-care-pd-rf-tcn/video-stability/straight-followup/availability.json'
STORE=Path('D:/gait-research-data/D_P/camera-cadence-20261101')
METHODS=['RGB2D','RGB3D','ProjectedReference2D','Reference3D']
FREQUENCIES=.4+np.arange(281)*.005
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def record(sid,cam):return next(r for r in json.loads((SOURCE/'results.json').read_text())['results'] if r['subject']==sid and r['camera']==cam and r['variant']=='heavy' and r['input_width']==800)
def load(sid,cam):
    path=SOURCE/f'subject-{sid:02d}/heavy/{cam}_800.npz';reference=path.with_name(f'{cam}_800_comparison.npz');assert sha(path)==record(sid,cam)['prediction_sha256'];assert sha(reference)==json.loads(AUDIT.read_text())['source_sha256'][str(reference)]
    p=np.load(path);r=np.load(reference);ix=np.searchsorted(p['video_frame_one_based'],r['frames']);np.testing.assert_array_equal(p['video_frame_one_based'][ix],r['frames']);lm=p['landmarks'][ix];world=p['world_landmarks'][ix]
    good=(p['pose_counts'][ix]>0)&np.isfinite(lm).all((1,2))&np.isfinite(world).all((1,2))&~r['reference_gap'].any(1)
    return dict(frames=r['frames'],lm=lm,world=world,reference=r['reference_3d'],projected=r['reference_xy'],good=good,files={str(z):sha(z) for z in [path,reference]})
def prepare():
    people=[]
    rows=json.loads((SOURCE/'results.json').read_text())['results'];subjects=sorted(r['subject'] for r in rows if r['camera']=='PG1' and r['variant']=='heavy' and r['input_width']==800)
    for sid in subjects:
        a,b=load(sid,'PG1'),load(sid,'PG2');np.testing.assert_array_equal(a['frames'],b['frames']);mask=a['good']&b['good'];starts={}
        for seconds in [2,4,6]:
            n=seconds*30;starts[str(seconds)]=next((i for i in range(len(mask)-n+1) if mask[i:i+n].all() and np.all(np.diff(a['frames'][i:i+n])==1)),None)
        people.append(dict(subject=sid,available_frames=len(mask),starts=starts,files={**a['files'],**b['files']}))
    selected=[r for r in people if r['starts']['4'] is not None];assert len(people)==30
    save(HERE/'availability.json',dict(people=people,available_counts={str(s):sum(r['starts'][str(s)] is not None for r in people) for s in [2,4,6]},selected_people=[r['subject'] for r in selected],rule='Earliest4sec common finite RGB and local3D pose,no reference gap,consecutive actualframes;2sec is firsthalf ofsame4sec. No output/score selection,no straightness restriction.'))
    save(HERE/'protocol.json',dict(code_sha256=sha(__file__),sources={str(z):sha(z) for z in [SOURCE/'results.json',AUDIT,HERE/'availability.json']},
      hypothesis='After limited synthetic robustness gains, test if cadence-period information can actually be extracted consistently across fixed-camera views, and if4sec helps relative to2sec on samepeople.',
      inputs='Existing RGB MediaPipeHeavy image and localworld landmarks;no height/floor scale/intrinsics needed for period. Localworld is RGBprediction,not reference3D. ProjectedReference2D is evaluator geometry control,not deployable input.',
      reference='Provider3Dleft-minus-right ankle period on the4sec window,computed by samefixed spectral algorithm. Derivedreference proxy only,no expert annotated footstrikes or validated cadence truth. Frequencyambiguity/turns may affect it. NoSPPB/TUGlabels,normal scores or clinical claims.',
      algorithm='For ankle difference subtract lineartrend,leading principal component,evaluate sine/cosine with constant+linear nuisance at281frequencies.4..1.8Hzstep.005. Pickminimum residual rounded12digits/tie lowerfreq;strideperiodfrequency*120 yields steps/min assumption. No model fit/calibration/hyperparameter search. Quality: harmonicR2>=.5,>=1.5estimatedcycles,and>.02Hz from bounds. Raw/qualitycoverage bothreported;unreliable is not low score.',
      people=f'{len(selected)}samepeople2/4sec nestedwindows,bothPG1/PG2,providerwalking maycontainturns;differentfromold14straightpeople. Allpeopleusedforsignalmeasurement,no training/new5seeds. ExistingMobilise-Dpeople/bouts/5seedsunchanged.',
      evaluation='Same4secReference3Dproxy for2/4sec comparisons. Referencequality andcandidatecoverage separate. Report raw/accepted meanabsolute discrepancy andbias toproxy,pairedviewdifference,andmatched2/4length oncommonreferenceacceptedpeople. Not groundtruth cadence MAE.',
      success='Exploratory4secRGB3Dtarget:bothviews quality>=80% ofselectedpeople,proxy discrepancy<=5steps/min ineachview,andpairedacceptedviewdifference<=5steps/min. Report components/counts,do not relaxthresholds. No clinical standard.',
      expected=dict(main_people=len(selected),main_tasks=len(selected)*2,estimates=len(selected)*16,pilot_people=2,pilot_tasks=4,pilot_estimates=32,metric_rows=12,paired_rows=6,frame_assignments=len(selected)*360,unique_video_frames=len(selected)*240,model_fits=0),
      verification='Source/codeSHA and exactframe/person/viewlink,reference equality,independentSVDprincipalcomponent+harmony scan,frequencygrid exact/R2tol1e-9,summary/personcounts independent. Pilotfirst2peoplethenmain. FrozeninputarraysD,smallcode/reportsC.',
      limits='Samealgorithmreferencecanoverstateagreement;no event-labelvalidation,no newclinic/phone/generalpathologytest.2/4secperiodnot30–60secmeancadence definition. Onegeometryselectedwindowperperson,qualityselectionmayfavorbetterdetectedpeople.',
      resources='CPU1,no newimageinference/GPU/download,estimate2-5min,arrays<10MBD;smalltablesC.',storage=dict(small=str(HERE),large=str(STORE))))
def tasks(mode):
    d=json.loads((HERE/'availability.json').read_text());people=d['selected_people'] if mode=='main' else d['selected_people'][:2];return [(sid,n) for sid in people for n in [2,4]]
def estimate(vector):
    t=np.arange(len(vector))/30.;t-=t.mean();base=np.c_[np.ones(len(t)),t];centered=vector-base@np.linalg.lstsq(base,vector,rcond=None)[0];cov=centered.T@centered;_,u=np.linalg.eigh(cov);pc=centered@u[:,-1];std=pc.std()
    if std<1e-12:return dict(cadence=None,r2=0.,cycles=0.,quality=False,frequency_index=-1)
    pc=pc/std;ss=float(pc@pc);losses=[]
    for f in FREQUENCIES:
        design=np.c_[base,np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t)];error=pc-design@np.linalg.lstsq(design,pc,rcond=None)[0];losses.append(float(error@error/ss))
    i=min(range(len(losses)),key=lambda j:(round(losses[j],12),j));f=FREQUENCIES[i];r2=1-losses[i];cycles=f*(len(t)-1)/30.;good=r2>=.5 and cycles>=1.5 and f>.42 and f<1.78
    return dict(cadence=float(f*120),r2=float(r2),cycles=float(cycles),quality=bool(good),frequency_index=int(i))
def work(task):
    sid,seconds=task;tag=f's{sid:02d}-{seconds}sec';ledger=STORE/f'{tag}.json'
    if ledger.exists():
        q=json.loads(ledger.read_text());assert q['code_sha256']==sha(__file__);return q
    chosen=next(r for r in json.loads((HERE/'availability.json').read_text())['people'] if r['subject']==sid);start=chosen['starts']['4'];n=seconds*30;rows=[];arrays={};files={}
    for cam in ['PG1','PG2']:
        d=load(sid,cam);sl=slice(start,start+n);files.update(d['files']);frames=d['frames'][sl];lm=d['lm'][sl];world=d['world'][sl];reference=d['reference'][sl];projected=d['projected'][sl]
        arrays.update({cam+'_frames':frames,cam+'_lm':lm,cam+'_world':world,cam+'_reference':reference,cam+'_projected':projected})
        vectors={'RGB2D':(lm[:,27,:2]-lm[:,28,:2])*[800.,600.],'RGB3D':world[:,27,:3]-world[:,28,:3],'ProjectedReference2D':projected[:,10]-projected[:,11],'Reference3D':reference[:,10]-reference[:,11]}
        for method,v in vectors.items():rows.append(dict(subject=sid,seconds=seconds,camera=cam,method=method,**estimate(v)))
    dest=STORE/f'{tag}.npz';np.savez_compressed(dest,**arrays);out=dict(task=list(task),rows=rows,sources=files,array_path=str(dest),array_sha256=sha(dest),code_sha256=sha(__file__));save(ledger,out);return out
def run(mode):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for q,h in protocol['sources'].items():assert sha(q)==h
    start=time.time();STORE.mkdir(parents=True,exist_ok=True);rows=[];ts=tasks(mode)
    for i,t in enumerate(ts,1):rows.extend(work(t)['rows']);print('PERIOD',i,len(ts),t,flush=True)
    pd.DataFrame(rows).to_csv(HERE/f'{mode}-estimates.csv',index=False);save(HERE/f'{mode}-completion.json',dict(tasks=len(ts),people=len(ts)//2,estimates=len(rows),code_sha256=sha(__file__),seconds=time.time()-start))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    else:run(a.mode)
