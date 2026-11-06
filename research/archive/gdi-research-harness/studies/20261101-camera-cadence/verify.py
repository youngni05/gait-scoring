from study import *
def independent(vector):
    timevec=np.linspace(0,(len(vector)-1)/30.,len(vector));timevec-=timevec.mean();base=np.stack([np.ones(len(vector)),timevec],1);residual=vector-base.dot(np.linalg.lstsq(base,vector,rcond=None)[0]);_,_,vt=np.linalg.svd(residual,full_matrices=False);signal=residual.dot(vt[0]);sd=signal.std()
    if sd<1e-12:return None,0.,False,-1
    signal/=sd;scores=[]
    for f in FREQUENCIES:
        x=np.column_stack([base,np.sin(timevec*2*np.pi*f),np.cos(timevec*2*np.pi*f)]);r=signal-x.dot(np.linalg.lstsq(x,signal,rcond=None)[0]);scores.append(float(np.sum(r*r)/np.sum(signal*signal)))
    i=min(range(len(scores)),key=lambda j:(round(scores[j],12),j));f=FREQUENCIES[i];r2=1-scores[i];quality=r2>=.5 and f*(len(vector)-1)/30.>=1.5 and f>.42 and f<1.78
    return f*120,r2,quality,i
def run(mode):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for q,h in protocol['sources'].items():assert sha(q)==h
    ts=tasks(mode);table=pd.read_csv(HERE/f'{mode}-estimates.csv');assert len(table)==len(ts)*8 and not table.duplicated(['subject','seconds','camera','method']).any();worst=0.
    for sid,seconds in ts:
        q=json.loads((STORE/f's{sid:02d}-{seconds}sec.json').read_text());assert q['code_sha256']==sha(HERE/'study.py') and sha(q['array_path'])==q['array_sha256'];z=np.load(q['array_path']);chosen=next(r for r in json.loads((HERE/'availability.json').read_text())['people'] if r['subject']==sid);start=chosen['starts']['4']
        for p,h in q['sources'].items():assert sha(p)==h
        np.testing.assert_array_equal(z['PG1_reference'],z['PG2_reference']);np.testing.assert_array_equal(z['PG1_frames'],z['PG2_frames']);assert np.all(np.diff(z['PG1_frames'])==1)
        for cam in ['PG1','PG2']:
            original=load(sid,cam);sl=slice(start,start+seconds*30);assert original['good'][sl].all()
            for a,b in [('frames','frames'),('lm','lm'),('world','world'),('reference','reference'),('projected','projected')]:np.testing.assert_array_equal(z[cam+'_'+a],original[b][sl])
            lm=z[cam+'_lm'];world=z[cam+'_world'];ref=z[cam+'_reference'];projected=z[cam+'_projected'];vectors=[(lm[:,27,:2]-lm[:,28,:2])*[800.,600.],world[:,27,:3]-world[:,28,:3],projected[:,10]-projected[:,11],ref[:,10]-ref[:,11]]
            for method,vector in zip(METHODS,vectors):
                cadence,r2,quality,index=independent(vector);r=table[table.subject.eq(sid)&table.seconds.eq(seconds)&table.camera.eq(cam)&table.method.eq(method)].iloc[0];assert int(r.frequency_index)==index and bool(r.quality)==quality
                if cadence is None:assert pd.isna(r.cadence)
                else:assert abs(cadence-r.cadence)<1e-12
                err=abs(r2-r.r2);assert err<1e-9;worst=max(worst,err)
    if mode=='main':
        d=table;reference=d[d.method.eq('Reference3D')&d.seconds.eq(4)&d.camera.eq('PG1')].set_index('subject');m=pd.read_csv(HERE/'metrics.csv');pairs=pd.read_csv(HERE/'paired.csv');length=pd.read_csv(HERE/'length-matched.csv');assert (len(m),len(pairs),len(length))==(12,6,6)
        for _,r in m.iterrows():
            g=d[d.method.eq(r.method)&d.seconds.eq(r.seconds)&d.camera.eq(r.camera)].set_index('subject');ok=reference.quality&g.quality&reference.cadence.notna()&g.cadence.notna();error=g.cadence-reference.cadence
            assert int(ok.sum())==r.accepted_people and int(g.quality.sum())==r.candidate_quality_people;expected=abs(error[ok]).mean();assert (pd.isna(expected) and pd.isna(r.accepted_proxy_discrepancy)) or abs(expected-r.accepted_proxy_discrepancy)<1e-10
        for _,r in pairs.iterrows():
            a=d[d.method.eq(r.method)&d.seconds.eq(r.seconds)&d.camera.eq('PG1')].set_index('subject');b=d[d.method.eq(r.method)&d.seconds.eq(r.seconds)&d.camera.eq('PG2')].set_index('subject');ok=reference.quality&a.quality&b.quality&a.cadence.notna()&b.cadence.notna();diff=b.cadence-a.cadence;assert int(ok.sum())==r.accepted_people;expected=abs(diff[ok]).mean();assert (pd.isna(expected) and pd.isna(r.accepted_view_difference)) or abs(expected-r.accepted_view_difference)<1e-10
        for _,r in length.iterrows():
            a=d[d.method.eq(r.method)&d.seconds.eq(2)&d.camera.eq(r.camera)].set_index('subject');b=d[d.method.eq(r.method)&d.seconds.eq(4)&d.camera.eq(r.camera)].set_index('subject');ok=reference.quality&a.quality&b.quality&a.cadence.notna()&b.cadence.notna();assert int(ok.sum())==r.common_people;diff=abs(b.cadence-reference.cadence)-abs(a.cadence-reference.cadence);expected=diff[ok].mean();assert (pd.isna(expected) and pd.isna(r.difference_4minus2)) or abs(expected-r.difference_4minus2)<1e-10
    save(HERE/f'{mode}-verification.json',dict(passed=True,tasks=len(ts),people=len(ts)//2,estimates=len(table),independent_period_replays=len(table),max_r2_difference=worst,r2_tolerance=1e-9,frequency_grid_exact=True,study_sha256=sha(HERE/'study.py'),verify_sha256=sha(__file__),analysis_sha256=sha(HERE/'analyze.py')));print('VERIFIED',mode,len(table),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
