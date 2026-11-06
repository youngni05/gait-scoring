from study import *
def independent_peaks(x):
    # Equivalent local-max/plateau candidates, followed by height-priority separation.
    candidates=[];i=1
    while i<len(x)-1:
        if x[i]>x[i-1]:
            end=i
            while end+1<len(x) and x[end+1]==x[i]:end+=1
            if end<len(x)-1 and x[end]>x[end+1]:candidates.append((i+end)//2)
            i=end
        i+=1
    candidates=np.array([i for i in candidates if x[i]>=x.mean()],dtype=int);keep=[]
    for i in candidates[np.argsort(x[candidates])[::-1]]:
        if all(abs(int(i)-j)>=48 for j in keep):keep.append(int(i))
    return sorted(keep)
def synthetic_check():
    index=np.arange(601);t=index/120.;points=np.zeros((len(t),6,3));points[:,:,0]=(.8*t)[:,None];points[:,0:2,0]+=.2;points[:,[0,2],1]=.1;points[:,[1,3],1]=-.1;points[:,4,0]+=.3*np.cos(2*np.pi*t/1.2);points[:,5,0]+=.3*np.cos(2*np.pi*t/1.2+np.pi);q,_=derive(points,np.zeros((len(t),6),bool),index,60,540);assert q['quality'] and abs(q['cadence']-100)<1e-10 and q['event_count']==7
def run(mode):
    synthetic_check();protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for p,h in protocol['sources'].items():assert sha(p)==h
    table=pd.read_csv(HERE/f'{mode}-events.csv');ids=tasks(mode);assert len(table)==len(ids) and table.subject.tolist()==ids
    for sid in ids:
        q=json.loads((STORE/f's{sid:02d}.json').read_text());assert q['code_sha256'] in protocol['accepted_output_code_sha256'] and q['array_sha256']==sha(q['array_path'])
        for p,h in q['sources'].items():assert sha(p)==h
        z=np.load(q['array_path']);m=loadmat(f'D:/gait-research-data/MoVi/reference/F_v3d_Subject_{sid}.mat',simplify_cells=True)[f'Subject_{sid}_F']['move'];names=[str(v) for v in m['markerName']];cols=[names.index(n) for n in NAMES];np.testing.assert_array_equal(z['source_columns'],cols);np.testing.assert_array_equal(z['points'],m['markerLocation'][z['indices']][:,cols]/1000.);np.testing.assert_array_equal(z['gaps'],m['markerGaps'][z['indices']][:,cols].astype(bool));assert q['start']==(int(z['frames'][0])-1)*4 and q['end']==(int(z['frames'][-1])-1)*4
        points=z['points'];back=(points[:,2,:2]+points[:,3,:2])/2;front=(points[:,0,:2]+points[:,1,:2])/2;axis=front-back;norm=np.sqrt(np.sum(axis**2,axis=1));forward=np.divide(axis,norm[:,None],out=np.zeros_like(axis),where=norm[:,None]>1e-6);signals=np.c_[np.sum((points[:,4,:2]-back)*forward,1),np.sum((points[:,5,:2]-back)*forward,1)];np.testing.assert_allclose(signals,z['signals'],rtol=0,atol=1e-10)
        valid=~z['gaps'].any(1)&np.isfinite(points).all((1,2))&(norm>1e-6);np.testing.assert_array_equal(valid,z['valid']);events=[];edges=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(int)))
        for a,b in zip(edges[::2],edges[1::2]):
            if b-a<9:continue
            for side in range(2):
                smooth=savgol_filter(signals[a:b,side],9,2,mode='interp');np.testing.assert_allclose(smooth,z['smoothed'][a:b,side],rtol=0,atol=1e-10)
                for i in independent_peaks(smooth):
                    frame=int(z['indices'][a+i])
                    if q['start']<=frame<=q['end']:events.append([frame,'L' if side==0 else 'R'])
        events=sorted(events,key=lambda e:(e[0],e[1]));assert events==q['events'];delta=np.diff([e[0] for e in events])/120.;value=60/np.median(delta) if len(delta) and (delta>0).all() else None
        if value is None:assert q['cadence'] is None
        else:assert abs(value-q['cadence'])<1e-10
        window=(z['indices']>=q['start'])&(z['indices']<=q['end']);alt=np.mean([a[1]!=b[1] for a,b in zip(events[:-1],events[1:])]) if len(events)>1 else 0.;quality=bool(valid[window].all() and len(events)>=3 and alt>=.8 and (delta>=.2).all() and (delta<=1.5).all());assert quality==q['quality'];assert abs(float(alt)-q['alternation'])<1e-12 and abs(valid[window].mean()-q['marker_valid_fraction'])<1e-12
        row=table[table.subject.eq(sid)].iloc[0];assert row.event_count==len(events) and bool(row.quality)==quality
        if value is not None:assert abs(row.cadence-value)<1e-10
    if mode=='main':
        d=pd.read_csv(HERE/'comparisons.csv');metrics=pd.read_csv(HERE/'metrics.csv');pairs=pd.read_csv(HERE/'paired.csv');assert (len(d),len(metrics),len(pairs))==(135,5,2)
        old=pd.read_csv(REFERENCE/'main-estimates.csv');old=old[old.seconds.eq(4)];evt=table.set_index('subject')
        for _,r in d.iterrows():
            a=old[old.subject.eq(r.subject)&old.method.eq(r.method)&old.camera.eq(r.camera)].iloc[0];e=evt.loc[r.subject];assert abs(r.cadence-a.cadence)<1e-10;both=bool(e.quality and a.quality and pd.notna(e.cadence) and pd.notna(a.cadence));assert both==r.both_quality
            if pd.notna(e.cadence):assert abs(r.discrepancy-(a.cadence-e.cadence))<1e-10
            half=bool(both and abs(a.cadence-e.cadence/2)<=5 and abs(a.cadence-e.cadence)>5);assert bool(r.half_proxy_flag)==half
        for _,r in metrics.iterrows():
            g=d[d.method.eq(r.method)&d.camera.eq(r.camera)];ok=g.both_quality;assert int(ok.sum())==r.both_quality_people and int(g.half_proxy_flag.sum())==r.half_proxy_people;value=abs(g.discrepancy[ok]).mean();assert (pd.isna(value) and pd.isna(r.mean_absolute_discrepancy)) or abs(value-r.mean_absolute_discrepancy)<1e-10
        for _,r in pairs.iterrows():
            g=d[d.method.eq(r.method)];a=g[g.camera.eq('PG1')].set_index('subject');b=g[g.camera.eq('PG2')].set_index('subject');ok=a.both_quality&b.both_quality;value=abs(b.cadence[ok]-a.cadence[ok]).mean();assert int(ok.sum())==r.both_camera_event_quality_people;assert (pd.isna(value) and pd.isna(r.mean_absolute_view_difference)) or abs(value-r.mean_absolute_view_difference)<1e-10
    save(HERE/f'{mode}-verification.json',dict(passed=True,tasks=len(ids),people=len(ids),event_replays=len(ids),synthetic100spm_check=True,event_indices_exact=True,tolerance=1e-10,verify_sha256=sha(__file__),analysis_sha256=sha(HERE/'analyze.py')));print('VERIFIED',mode,len(ids),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
