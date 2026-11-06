"""Complete the original partial audit without modifying calculation code or outputs."""
from study import *
def close(a,b):
    assert (pd.isna(a) and pd.isna(b)) or abs(a-b)<=1e-10,(a,b)
def run(mode):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    checked={}
    def check(path,digest):
        path=str(path)
        if path not in checked:checked[path]=sha(path)
        assert checked[path]==digest,path
    for path,h in protocol['sources'].items():check(path,h)
    # This verifier is separate from the frozen main calculation revision.
    scope={'np':np};tree=ast.parse((gate.HERE/'verify.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='independent');exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_manual_gate','exec'),scope)
    people=gate.source.ids(mode);clips=json.loads(gate.source.CLIPS.read_text());d=pd.read_csv(HERE/f'{mode}-components.csv');metrics=pd.read_csv(HERE/f'{mode}-metrics.csv');paired=pd.read_csv(HERE/f'{mode}-paired.csv');names=['NoGate','VisibilityOnly','OrientationOnly','Combined'];expected={(sid,condition,cam,name) for sid in people for condition in ['straight','turn'] for cam in ['PG1','PG2'] for name in names};assert set(map(tuple,d[['subject','condition','camera','component']].values))==expected and len(d)==len(expected);replays=0
    gateprotocol=json.loads((gate.HERE/'protocol.json').read_text())
    for path,h in gateprotocol['sources'].items():check(path,h)
    source=gate.source;parentprotocol=json.loads((source.HERE/'protocol.json').read_text())
    for path,h in parentprotocol['sources'].items():check(path,h)
    check(source.HERE/'study.py',parentprotocol['code_sha256']);check(gate.HERE/'study.py',gateprotocol['code_sha256'])
    for sid in people:
        receipt=gate.STORE/f's{sid:02d}.json';q=json.loads(receipt.read_text());check(gate.HERE/'study.py',q['code_sha256']);check(q['array_path'],q['array_sha256'])
        for path,h in q['sources'].items():check(path,h)
        z=np.load(q['array_path']);parent=source.STORE/f's{sid:02d}.json';check(parent,q['sources'][str(parent)]);raw=json.loads(parent.read_text());check(source.HERE/'study.py',raw['code_sha256'])
        for segment in raw['segments']:
            condition=segment['condition'];check(segment['array_path'],segment['array_sha256'])
            for path,h in segment['sources'].items():check(path,h)
            a=np.load(segment['array_path'])
            for cam in ['PG1','PG2']:
                tag=condition+'_'+cam;clip=next(x for x in clips if x['subject']==sid and x['condition']==condition and x['camera']==cam);frames=np.arange(clip['frames_one_based'][0],clip['frames_one_based'][1]+1);assert len(frames)==60;np.testing.assert_array_equal(z[tag+'_frames'],frames)
                original=source.ref.load(sid,cam);ix=np.searchsorted(original['frames'],frames);np.testing.assert_array_equal(original['frames'][ix],frames);assert original['good'][ix].all()
                for x,y in [('world','world'),('image','lm'),('frames','frames')]:np.testing.assert_array_equal(z[tag+'_'+x],a[cam+'_'+y]);np.testing.assert_array_equal(z[tag+'_'+x],original[y][ix])
                world,image=z[tag+'_world'],z[tag+'_image'];combined,span,valid=scope['independent'](world,image);orientation=span is not None and span<=30;visibility=all(np.isfinite(world[:,j,:3]).all() and np.isfinite(image[:,j,:]).all() and all(float(v)>=.5 for v in image[:,j,3]) for j in [11,12,23,24]);assert visibility==valid;decision=dict(NoGate=True,VisibilityOnly=visibility,OrientationOnly=orientation,Combined=combined);assert combined==bool(orientation and visibility)
                minimum=min(float(image[t,j,3]) for t in range(60) for j in [11,12,23,24]);g=d[d.subject.eq(sid)&d.condition.eq(condition)&d.camera.eq(cam)]
                for _,r in g.iterrows():assert bool(r.accepted)==bool(decision[r.component]);close(r.span,span if span is not None else np.nan);close(r.min_visibility,minimum)
                replays+=1
    assert len(metrics)==8 and not metrics.duplicated(['component','camera']).any()
    records=d.to_dict('records')
    for _,r in metrics.iterrows():
        a=[x for x in records if x['component']==r.component and x['camera']==r.camera and x['condition']=='straight'];b=[x for x in records if x['component']==r.component and x['camera']==r.camera and x['condition']=='turn'];assert len(a)==len(b)==len(people)==r.people;sa=sum(bool(x['accepted']) for x in a);ta=sum(bool(x['accepted']) for x in b);assert sa==r.straight_accepted and ta==r.turn_accepted;close(sa/len(a),r.straight_retention);close(ta/len(b),r.turn_false_acceptance)
    assert len(paired)==8 and not paired.duplicated(['component','condition']).any()
    for _,r in paired.iterrows():
        g=[x for x in records if x['component']==r.component and x['condition']==r.condition];a={x['subject']:bool(x['accepted']) for x in g if x['camera']=='PG1'};b={x['subject']:bool(x['accepted']) for x in g if x['camera']=='PG2'};assert set(a)==set(b)==set(people);assert r.people==len(people) and r.both_accepted==sum(a[s] and b[s] for s in people) and r.disagree==sum(a[s]!=b[s] for s in people)
    if mode=='main':
        process=json.loads((HERE/'main-process.json').read_text());exit=json.loads((HERE/'main-process-exit.json').read_text());assert process['pid']==exit['pid'] and exit['exit_code']==0
    save(HERE/f'{mode}-full-verification.json',dict(passed=True,people=len(people),tasks=replays,component_rows=len(d),metric_rows=len(metrics),paired_rows=len(paired),independent_gate_replays=replays,independently_checked_files=len(checked),source_sha256=checked,original_code_sha256=sha(HERE/'study.py'),verify_sha256=sha(__file__),span_tolerance=1e-10,new_model_fits=0));print('FULL_VERIFIED',mode,replays,len(d),len(metrics),len(paired),'files',len(checked),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
