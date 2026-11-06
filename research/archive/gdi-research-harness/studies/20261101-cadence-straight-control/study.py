"""Frozen straight/turn windows: audit cadence proxies, without claiming contact truth."""
from pathlib import Path
import importlib.util,json,argparse,time
HERE=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
audit=module('heel_audit_fixed',HERE.parent/'20261101-cadence-reference-audit/study.py')
ref=audit.ref;np,pd,sha,save=ref.np,ref.pd,ref.sha,ref.save
STORE=Path('D:/gait-research-data/D_P/cadence-straight-control-20261101')
CLIPS=HERE.parent/'20261031-care-pd-rf-tcn/video-stability/straight-followup/clips.json'
CONDITIONS=['straight','turn']
def ids(mode):
    people=sorted({r['subject'] for r in json.loads(CLIPS.read_text())});assert len(people)==14
    return people if mode=='main' else people[:2]
def prepare():
    sources=[CLIPS,CLIPS.with_name('protocol.json'),CLIPS.with_name('availability.json'),Path(audit.__file__),Path(audit.__file__).with_name('verify.py'),Path(ref.__file__),Path(ref.__file__).with_name('verify.py')]
    save(HERE/'protocol.json',dict(code_sha256=sha(__file__),sources={str(p):sha(p) for p in sources},
      hypothesis='At equal2sec duration in the same14people, disagreement between spectral and optical-event proxies may be smaller during preselected straight walking than turning. Diagnose geometric assumptions,not retune a proxy until it agrees.',
      selection='Reuse all56 clips previously frozen before cadence outcomes:14people x straight/turn x PG1/PG2. Source geometry selected by reference visibility/heading/path,never by current cadence. Same actual frames bothviews. No reselection or person exclusion by disagreement.',
      inputs='Cached RGB image/local3D ankle difference;ProjectedReference2D evaluator-only control;Reference3D spectral proxy. No reference coordinates enter RGB inputs. Raw6physicalmarkers feed only optical event comparator.',
      algorithms='Import frozen camera-cadence estimate() and cadence-reference-audit derive() unchanged. Spectral281frequencygrid .4..1.8Hz,2steps/stride;qualityR2>=.5,cycles>=1.5,boundarymargin.02. Optical120Hz heelAPrelativeposteriorwaist/dynamicpelvicaxis,SavGol9degree2,positivepeaks48samplesapart;quality allvalid,>=3events,alternation>=.8,interval.2..1.5. Opticalcontext±.5sec clippedproviderwalking,RGBexact60frames.',
      primary='Withinperson Reference3D raw finite spectral-vs-event absolute discrepancy contrast straightminus turn,among event-quality people available in BOTHconditions. Raw finite spectral primary avoids structural2sec1.5cycle selection. Original quality rules retained and all/quality counts reported separately,not relaxed.',
      success='Exploratory consistency target,not clinical:>=10pairedpeople,straight mean absolute proxy discrepancy<=5steps/min and improvement>=5steps/min vs turns. Components reported even if joint criterion fails. Does not adjudicate true contacts.',
      limits='Both quantities are proxies without expert/force contacts. Dynamicwaist-axis event adaptation unvalidated.2sec spectral resolution and quality gate exclude estimated<91.525spm;report raw and gated. Same14people differ from previous27;within14 contrast only. NoSPPB/TUGlabels,phonevalidation,newscoretraining/seeds. Original scorepeople/bouts/5seeds unchanged.',
      expected=dict(main_tasks=14,pilot_tasks=2,people=14,event_segments=28,period_estimates=224,detail_rows=140,metric_rows=10,geometry_pairs=5,view_pairs=4,new_model_fits=0),
      verification='Source/code/arraysSHA,rawMAT/12joint/frame mapping,independent manualpeak and SVD spectral replays;event indices exact,numerics1e-10,R2 1e-9,gridexact. Independently recompute all tables. Pilot2people then main.',
      resources='CPU1,2-5min,noGPU/newinference/download;selectedarrays<10MB D,code/reportsC',storage=dict(small=str(HERE),large=str(STORE))))
def vectors(z,cam):
    lm=z[cam+'_lm'];world=z[cam+'_world'];r=z[cam+'_reference'];p=z[cam+'_projected']
    return {'RGB2D':(lm[:,27,:2]-lm[:,28,:2])*[800.,600.],'RGB3D':world[:,27,:3]-world[:,28,:3],'ProjectedReference2D':p[:,10]-p[:,11],'Reference3D':r[:,10]-r[:,11]}
def work(sid):
    dest=STORE/f's{sid:02d}.json'
    if dest.exists():
        q=json.loads(dest.read_text());assert q['code_sha256']==sha(__file__);return q
    clips=json.loads(CLIPS.read_text());path=Path(f'D:/gait-research-data/MoVi/reference/F_v3d_Subject_{sid}.mat');m=audit.loadmat(path,simplify_cells=True)[f'Subject_{sid}_F']['move'];names=[str(x) for x in m['markerName']];cols=[names.index(n) for n in audit.NAMES];jointcols=[names.index(n) for n in ['LSJC','RSJC','LEJC','REJC','LWJC','RWJC','LHIP','RHIP','LKJC','RKJC','LAJC','RAJC']]
    np.testing.assert_array_equal(m['flags120'].astype(int),(m['flags30'].astype(int)-1)*4+1)
    walk=list(m['motions_list']).index('walking');lo,hi=m['flags120'][walk].astype(int)-1;segments=[]
    for condition in CONDITIONS:
        arrays={};sources={str(path):sha(path),str(CLIPS):sha(CLIPS)};rows=[]
        for cam in ['PG1','PG2']:
            clip=next(x for x in clips if x['subject']==sid and x['camera']==cam and x['condition']==condition);d=ref.load(sid,cam);frames=np.arange(clip['frames_one_based'][0],clip['frames_one_based'][1]+1);ix=np.searchsorted(d['frames'],frames);np.testing.assert_array_equal(d['frames'][ix],frames);assert len(frames)==60 and d['good'][ix].all();sources.update(d['files'])
            for k in ['frames','lm','world','reference','projected']:arrays[cam+'_'+k]=d[k][ix]
            np.testing.assert_array_equal(m['markerLocation'][(frames-1)*4][:,jointcols],arrays[cam+'_reference'])
            for method,v in vectors(arrays,cam).items():rows.append(dict(subject=sid,condition=condition,camera=cam,method=method,**ref.estimate(v)))
        np.testing.assert_array_equal(arrays['PG1_frames'],arrays['PG2_frames']);np.testing.assert_array_equal(arrays['PG1_reference'],arrays['PG2_reference'])
        start,end=(arrays['PG1_frames'][[0,-1]]-1)*4;indices=np.arange(max(lo,start-60),min(hi,end+60)+1);points=m['markerLocation'][indices][:,cols]/1000.;gaps=m['markerGaps'][indices][:,cols].astype(bool);event,derived=audit.derive(points,gaps,indices,int(start),int(end))
        array=STORE/f's{sid:02d}-{condition}.npz';np.savez_compressed(array,**arrays,points=points,gaps=gaps,indices=indices,source_columns=cols,**derived)
        segments.append(dict(subject=sid,condition=condition,start=int(start),end=int(end),**event,rows=rows,array_path=str(array),array_sha256=sha(array),sources=sources))
    q=dict(subject=sid,segments=segments,code_sha256=sha(__file__));save(dest,q);return q
def run(mode):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for p,h in protocol['sources'].items():assert sha(p)==h
    STORE.mkdir(parents=True,exist_ok=True);rows=[];events=[];started=time.time()
    for i,sid in enumerate(ids(mode),1):
        for q in work(sid)['segments']:
            rows+=q['rows'];events.append({k:v for k,v in q.items() if k not in ['rows','sources','events']})
        print('GEOMETRY',i,sid,flush=True)
    pd.DataFrame(rows).to_csv(HERE/f'{mode}-estimates.csv',index=False);pd.DataFrame(events).to_csv(HERE/f'{mode}-events.csv',index=False);save(HERE/f'{mode}-completion.json',dict(tasks=len(ids(mode)),people=len(ids(mode)),event_segments=len(events),period_estimates=len(rows),new_model_fits=0,code_sha256=sha(__file__),seconds=time.time()-started))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
