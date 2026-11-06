"""Fixed component ablation; diagnostic only, no post-hoc threshold optimization."""
from pathlib import Path
import importlib.util,json,argparse,ast,time,os,datetime
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixed_gate',HERE.parent/'20261101-camera-straight-gate/study.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
np,pd,sha,save=gate.np,gate.pd,gate.sha,gate.save
def prepare():
    files=[gate.HERE/n for n in ['study.py','verify.py','protocol.json','main-verification.json','main-gates.csv','README.md']]
    save(HERE/'protocol.json',dict(code_sha256=sha(__file__),sources={str(p):sha(p) for p in files},hypothesis='FixedCombinedgate retains straight1/14PG1 vs14/14PG2. Diagnose whether visibility or torsoorientation component drives the view disparity before changing any capture rule.',inputs='Same14people56frozenstraight/turn2secRGBclips. Reuse all originalarrays unchanged onD;no new bulkdata/training/inference.',conditions='NoGate;VisibilityOnly(original alljointfinite/imagefinite/visibility>=.5);OrientationOnly(original validaxis/span<=30);Combinedoriginal. All thresholds unchanged. Counterfactual component removals are new diagnostic comparators,not a relaxed passing criterion for oldexperiment.',evaluation='Percondition/view/component straightretention andturnfalseacceptance,pairedviewagreement,perclipspan/visibility. No component selected for deployment or tuned using outcomes.',success='Attribution diagnostic,not improvementclaim. Visibilitydominant if removing visibility restores>=10 of13PG1rejectedstraightclips while orientation threshold unchanged;orientationdominant if analogous visibilityonly restores>=10. Both/neither possible. Report all components and eachviewturnacceptance. Not clinical standard.',expected=dict(main_tasks=56,people=14,diagnostic_rows=224,metric_rows=8,paired_rows=8,pilot_clips=8,new_model_fits=0),verification='Frozen source/array/codeSHA andperson/frame/condition,originalCombined exactreplay. Independentmanualphaseunwrap andquantile from frozen verifier for56clips,componentboolean exact andspan1e-10. Independently count all tables. Pilot2people then main.',resources='CPU1<1min,noGPU/download/training,smalltablesC,reuseimmutableDarrays;originalscorepeople/bouts/5seeds unchanged',limits='Same14developmentpeople,proxyorientation andestimatedvisibility,no expertcontact/SPPB/TUGlabels. A counterfactual improvementdoesnotvalidate unseenphonecapture. No newthresholdgrid or promotion without independentmeasurement evidence.'))
def execute(mode):
    p=json.loads((HERE/'protocol.json').read_text());assert p['code_sha256']==sha(__file__)
    for path,h in p['sources'].items():assert sha(path)==h
    scope={'np':np};tree=ast.parse((gate.HERE/'verify.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='independent');exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_independent','exec'),scope)
    people=gate.source.ids(mode);old=pd.read_csv(gate.HERE/'main-gates.csv');rows=[];started=time.time()
    for sid in people:
        q=json.loads((gate.STORE/f's{sid:02d}.json').read_text());assert sha(q['array_path'])==q['array_sha256'];z=np.load(q['array_path'])
        for r in q['rows']:
            tag=r['condition']+'_'+r['camera'];world=z[tag+'_world'];image=z[tag+'_image'];same=gate.gate(world,image);ind,span,valid=scope['independent'](world,image);assert ind==same['accepted']==r['accepted'] and valid==same['landmarks_valid']==r['landmarks_valid'];assert (span is None and r['span'] is None) or abs(span-r['span'])<1e-10
            reference=old[old.subject.eq(sid)&old.condition.eq(r['condition'])&old.camera.eq(r['camera'])].iloc[0];assert bool(reference.accepted)==ind
            orientation=span is not None and span<=30;visibility=bool(np.isfinite(world[:,[11,12,23,24],:3]).all() and np.isfinite(image[:,[11,12,23,24],:]).all() and (image[:,[11,12,23,24],3]>=.5).all());assert valid==visibility
            for name,accepted in [('NoGate',True),('VisibilityOnly',visibility),('OrientationOnly',orientation),('Combined',ind)]:rows.append(dict(subject=sid,condition=r['condition'],camera=r['camera'],component=name,accepted=bool(accepted),span=span,min_visibility=float(image[:,[11,12,23,24],3].min())))
    d=pd.DataFrame(rows);metrics=[];paired=[]
    for (component,cam),g in d.groupby(['component','camera']):
        a=g[g.condition.eq('straight')];b=g[g.condition.eq('turn')];metrics.append(dict(component=component,camera=cam,people=len(a),straight_accepted=int(a.accepted.sum()),straight_retention=a.accepted.mean(),turn_accepted=int(b.accepted.sum()),turn_false_acceptance=b.accepted.mean()))
    for (component,condition),g in d.groupby(['component','condition']):
        a=g[g.camera.eq('PG1')].set_index('subject');b=g[g.camera.eq('PG2')].set_index('subject');paired.append(dict(component=component,condition=condition,people=len(a),both_accepted=int((a.accepted&b.accepted).sum()),disagree=int((a.accepted!=b.accepted).sum())))
    d.to_csv(HERE/f'{mode}-components.csv',index=False);m=pd.DataFrame(metrics);m.to_csv(HERE/f'{mode}-metrics.csv',index=False);pd.DataFrame(paired).to_csv(HERE/f'{mode}-paired.csv',index=False)
    # Independently sum persisted clip decisions, rather than trust table creation.
    persisted=pd.read_csv(HERE/f'{mode}-components.csv')
    for r in metrics:
        a=[x for x in rows if x['component']==r['component'] and x['camera']==r['camera'] and x['condition']=='straight'];b=[x for x in rows if x['component']==r['component'] and x['camera']==r['camera'] and x['condition']=='turn'];assert sum(x['accepted'] for x in a)==r['straight_accepted'] and sum(x['accepted'] for x in b)==r['turn_accepted']
    assert len(persisted)==len(people)*16
    save(HERE/f'{mode}-verification.json',dict(passed=True,people=len(people),tasks=len(people)*4,component_rows=len(d),metric_rows=len(metrics),paired_rows=len(paired),independent_gate_replays=len(people)*4,source_code_sha256=sha(gate.HERE/'verify.py'),code_sha256=sha(__file__),new_model_fits=0,seconds=time.time()-started));print('VERIFIED_COMPONENTS',mode,len(people)*4,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='pilot':execute('pilot')
    else:
        v=json.loads((HERE/'pilot-verification.json').read_text());assert v['passed'] and v['tasks']==8 and v['code_sha256']==sha(__file__);save(HERE/'main-process.json',dict(pid=os.getpid(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),entry='study.py main'));execute('main');save(HERE/'main-process-exit.json',dict(pid=os.getpid(),exit_code=0))
