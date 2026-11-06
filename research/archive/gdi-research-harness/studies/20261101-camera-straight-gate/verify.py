from study import *
def close(a,b):assert (pd.isna(a) and pd.isna(b)) or abs(a-b)<1e-10,(a,b)
def independent(world,image):
    import math
    good=all(np.isfinite(world[:,j,:3]).all() and np.isfinite(image[:,j,:]).all() and min(image[:,j,3])>=.5 for j in [11,12,23,24]);spans=[]
    for l,r in [(11,12),(23,24)]:
        angle=[]
        for i in range(len(world)):
            x=float(world[i,r,0]-world[i,l,0]);z=float(world[i,r,2]-world[i,l,2])
            if not math.isfinite(x+z) or math.hypot(x,z)<=1e-6:return False,None,False
            angle.append(math.atan2(z,x))
        # Independent sequential unwrap equivalent to numpy.unwrap's principal delta.
        u=[angle[0]]
        for before,after in zip(angle[:-1],angle[1:]):
            delta=after-before
            if delta>math.pi:delta-=2*math.pi
            elif delta<-math.pi:delta+=2*math.pi
            u.append(u[-1]+delta)
        values=sorted(u)
        def percentile(p):
            t=(len(values)-1)*p;k=int(t);return values[k] if k==len(values)-1 else values[k]+(values[k+1]-values[k])*(t-k)
        spans.append(math.degrees(percentile(.95)-percentile(.05)))
    span=max(spans);return bool(good and span<=30),span,bool(good)
def run(mode):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for p,h in protocol['sources'].items():assert sha(p)==h
    d=pd.read_csv(HERE/f'{mode}-gates.csv');assert len(d)==len(source.ids(mode))*4 and not d.duplicated(['subject','condition','camera']).any()
    for sid in source.ids(mode):
        q=json.loads((STORE/f's{sid:02d}.json').read_text());assert q['code_sha256']==sha(HERE/'study.py') and sha(q['array_path'])==q['array_sha256'];z=np.load(q['array_path']);parent=json.loads((source.STORE/f's{sid:02d}.json').read_text())
        for p,h in q['sources'].items():assert sha(p)==h
        for segment in parent['segments']:
            original=np.load(segment['array_path'])
            for cam in ['PG1','PG2']:
                tag=segment['condition']+'_'+cam
                for a,b in [('world','world'),('image','lm'),('frames','frames')]:np.testing.assert_array_equal(z[tag+'_'+a],original[cam+'_'+b])
                accepted,span,valid=independent(z[tag+'_world'],z[tag+'_image']);r=d[d.subject.eq(sid)&d.condition.eq(segment['condition'])&d.camera.eq(cam)].iloc[0];assert bool(r.accepted)==accepted and bool(r.landmarks_valid)==valid;close(span if span is not None else np.nan,r.span)
    if mode=='main':
        m=pd.read_csv(HERE/'metrics.csv');p=pd.read_csv(HERE/'paired.csv');c=pd.read_csv(HERE/'proxy-comparison.csv');assert (len(m),len(p),len(c))==(2,2,8)
        for _,r in m.iterrows():
            a=d[d.camera.eq(r.camera)&d.condition.eq('straight')];b=d[d.camera.eq(r.camera)&d.condition.eq('turn')];assert len(a)==r.straight_people==14 and len(b)==r.turn_people==14;assert a.accepted.sum()==r.straight_accepted and b.accepted.sum()==r.turn_accepted;close(a.accepted.mean(),r.straight_retention);close(b.accepted.mean(),r.turn_false_acceptance)
        for _,r in p.iterrows():
            g=d[d.condition.eq(r.condition)];a=g[g.camera.eq('PG1')].set_index('subject');b=g[g.camera.eq('PG2')].set_index('subject');assert r.people==14 and (a.accepted&b.accepted).sum()==r.both_accepted and (a.accepted!=b.accepted).sum()==r.one_view_disagrees and (~a.accepted&~b.accepted).sum()==r.both_rejected
        old=pd.read_csv(source.HERE/'comparisons.csv')
        for _,r in c.iterrows():
            g=old[old.condition.eq(r.condition)&old.method.eq(r.method)&old.camera.eq(r.camera)].set_index('subject');gate=d[d.condition.eq(r.condition)&d.camera.eq(r.camera)].set_index('subject').accepted;ok=g.raw_available&gate;assert len(g)==r.people==14 and gate.sum()==r.gate_accepted and g.raw_available.sum()==r.event_available and ok.sum()==r.gated_event_available and (ok&g.quality).sum()==r.gated_period_quality;close(abs(g.discrepancy[g.raw_available]).mean(),r.ungated_discrepancy);close(abs(g.discrepancy[ok]).mean(),r.gated_discrepancy)
    save(HERE/f'{mode}-verification.json',dict(passed=True,tasks=len(source.ids(mode)),people=len(source.ids(mode)),gate_replays=len(d),new_model_fits=0,span_tolerance=1e-10,verify_sha256=sha(__file__),study_sha256=sha(HERE/'study.py'),analysis_sha256=sha(HERE/'analyze.py')));print('VERIFIED',mode,len(d),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
