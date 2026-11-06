from study import *
import ast
def helper(path,name,scope):
    tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope);return scope[name]
period=helper(Path(ref.__file__).with_name('verify.py'),'independent',dict(np=np,FREQUENCIES=ref.FREQUENCIES))
peaks=helper(Path(audit.__file__).with_name('verify.py'),'independent_peaks',dict(np=np))
synthetic=helper(Path(audit.__file__).with_name('verify.py'),'synthetic_check',dict(np=np,derive=audit.derive))
def close(a,b,tol=1e-10):
    assert (pd.isna(a) and pd.isna(b)) or abs(a-b)<=tol,(a,b)
def run(mode):
    synthetic();protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for p,h in protocol['sources'].items():assert sha(p)==h
    periods=pd.read_csv(HERE/f'{mode}-estimates.csv');events=pd.read_csv(HERE/f'{mode}-events.csv');clips=json.loads(CLIPS.read_text());people=ids(mode)
    assert len(periods)==len(people)*16 and len(events)==len(people)*2
    assert not periods.duplicated(['subject','condition','camera','method']).any();assert not events.duplicated(['subject','condition']).any();replays=0
    for sid in people:
        q=json.loads((STORE/f's{sid:02d}.json').read_text());assert q['code_sha256']==sha(HERE/'study.py');m=audit.loadmat(f'D:/gait-research-data/MoVi/reference/F_v3d_Subject_{sid}.mat',simplify_cells=True)[f'Subject_{sid}_F']['move'];names=[str(x) for x in m['markerName']]
        for segment in q['segments']:
            condition=segment['condition'];assert sha(segment['array_path'])==segment['array_sha256'];z=np.load(segment['array_path'])
            for p,h in segment['sources'].items():assert sha(p)==h
            for cam in ['PG1','PG2']:
                clip=next(x for x in clips if x['subject']==sid and x['condition']==condition and x['camera']==cam);frames=np.arange(clip['frames_one_based'][0],clip['frames_one_based'][1]+1);np.testing.assert_array_equal(z[cam+'_frames'],frames);original=ref.load(sid,cam);ix=np.searchsorted(original['frames'],frames);assert original['good'][ix].all()
                for k in ['frames','lm','world','reference','projected']:np.testing.assert_array_equal(z[cam+'_'+k],original[k][ix])
                cols=[names.index(n) for n in ['LSJC','RSJC','LEJC','REJC','LWJC','RWJC','LHIP','RHIP','LKJC','RKJC','LAJC','RAJC']];np.testing.assert_array_equal(z[cam+'_reference'],m['markerLocation'][(frames-1)*4][:,cols])
                for method,vector in vectors(z,cam).items():
                    cadence,r2,quality,index=period(vector);row=periods[periods.subject.eq(sid)&periods.condition.eq(condition)&periods.camera.eq(cam)&periods.method.eq(method)].iloc[0];close(cadence if cadence is not None else np.nan,row.cadence,1e-12);close(r2,row.r2,1e-9);assert bool(row.quality)==quality and int(row.frequency_index)==index;replays+=1
            np.testing.assert_array_equal(z['PG1_frames'],z['PG2_frames']);cols=[names.index(n) for n in audit.NAMES];np.testing.assert_array_equal(z['source_columns'],cols);np.testing.assert_array_equal(z['points'],m['markerLocation'][z['indices']][:,cols]/1000.);np.testing.assert_array_equal(z['gaps'],m['markerGaps'][z['indices']][:,cols].astype(bool));assert segment['start']==(int(z['PG1_frames'][0])-1)*4 and segment['end']==(int(z['PG1_frames'][-1])-1)*4
            walk=list(m['motions_list']).index('walking');lo,hi=m['flags120'][walk].astype(int)-1;np.testing.assert_array_equal(z['indices'],np.arange(max(lo,segment['start']-60),min(hi,segment['end']+60)+1));np.testing.assert_array_equal(m['flags120'].astype(int),(m['flags30'].astype(int)-1)*4+1)
            points=z['points'];front=(points[:,0,:2]+points[:,1,:2])/2;back=(points[:,2,:2]+points[:,3,:2])/2;axis=front-back;norm=np.sqrt(np.sum(axis**2,1));forward=np.divide(axis,norm[:,None],out=np.zeros_like(axis),where=norm[:,None]>1e-6);signals=np.c_[np.sum((points[:,4,:2]-back)*forward,1),np.sum((points[:,5,:2]-back)*forward,1)];np.testing.assert_allclose(signals,z['signals'],rtol=0,atol=1e-10);valid=~z['gaps'].any(1)&np.isfinite(points).all((1,2))&(norm>1e-6);np.testing.assert_array_equal(valid,z['valid']);found=[];edges=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(int)))
            for a,b in zip(edges[::2],edges[1::2]):
                if b-a<9:continue
                for side in range(2):
                    smooth=audit.savgol_filter(signals[a:b,side],9,2,mode='interp');np.testing.assert_allclose(smooth,z['smoothed'][a:b,side],rtol=0,atol=1e-10)
                    for k in peaks(smooth):
                        frame=int(z['indices'][a+k])
                        if segment['start']<=frame<=segment['end']:found.append([frame,'L' if side==0 else 'R'])
            found=sorted(found,key=lambda x:(x[0],x[1]));assert found==segment['events'];delta=np.diff([x[0] for x in found])/120.;cadence=60/np.median(delta) if len(delta) and (delta>0).all() else np.nan;win=(z['indices']>=segment['start'])&(z['indices']<=segment['end']);alt=np.mean([a[1]!=b[1] for a,b in zip(found[:-1],found[1:])]) if len(found)>1 else 0.;quality=bool(valid[win].all() and len(found)>=3 and alt>=.8 and (delta>=.2).all() and (delta<=1.5).all());assert quality==segment['quality'];close(cadence,segment['cadence'] if segment['cadence'] is not None else np.nan);close(alt,segment['alternation']);close(valid[win].mean(),segment['marker_valid_fraction']);assert len(found)==segment['event_count']
            if valid[win].all():
                yaw=np.unwrap(np.arctan2(forward[win,1],forward[win,0]));span=np.rad2deg(np.percentile(yaw,95)-np.percentile(yaw,5));path=np.linalg.norm(np.diff(back[win],axis=0),axis=1).sum();distance=np.linalg.norm(back[win][-1]-back[win][0]);efficiency=distance/max(path,1e-12);speed=distance/((segment['end']-segment['start'])/120.);close(span,segment['heading_p90_span']);close(efficiency,segment['path_efficiency']);close(speed,segment['net_speed']);assert segment['straight']==bool(span<=30 and efficiency>=.95 and speed>=.2)
            else:assert not segment['straight']
            row=events[events.subject.eq(sid)&events.condition.eq(condition)].iloc[0];close(cadence,row.cadence);assert bool(row.quality)==quality and row.event_count==len(found) and bool(row.straight)==segment['straight']
    if mode=='main':
        d=pd.read_csv(HERE/'comparisons.csv');metrics=pd.read_csv(HERE/'metrics.csv');geometry=pd.read_csv(HERE/'geometry-paired.csv');views=pd.read_csv(HERE/'view-paired.csv');assert (len(d),len(metrics),len(geometry),len(views))==(140,10,5,4)
        for _,r in d.iterrows():
            p=periods[periods.subject.eq(r.subject)&periods.condition.eq(r.condition)&periods.method.eq(r.method)&periods.camera.eq(r.camera)].iloc[0];e=events[events.subject.eq(r.subject)&events.condition.eq(r.condition)].iloc[0];close(r.cadence,p.cadence);close(r.cadence_event,e.cadence);close(r.discrepancy,p.cadence-e.cadence);assert bool(r.raw_available)==bool(e.quality and pd.notna(p.cadence-e.cadence));assert bool(r.both_quality)==bool(r.raw_available and p.quality)
        for _,r in metrics.iterrows():
            g=d[d.condition.eq(r.condition)&d.method.eq(r.method)&d.camera.eq(r.camera)];assert len(g)==r.people;assert g.quality_event.sum()==r.event_quality and g.quality.sum()==r.period_quality and g.raw_available.sum()==r.raw_available and g.both_quality.sum()==r.both_quality and g.straight.sum()==r.actual_straight;close(abs(g.discrepancy[g.raw_available]).mean(),r.raw_discrepancy);close(abs(g.discrepancy[g.both_quality]).mean(),r.quality_discrepancy);close(g.discrepancy[g.raw_available].mean(),r.raw_bias)
        for _,r in geometry.iterrows():
            g=d[d.method.eq(r.method)&d.camera.eq(r.camera)];a=g[g.condition.eq('straight')].set_index('subject');b=g[g.condition.eq('turn')].set_index('subject');ok=a.raw_available&b.raw_available;aa=abs(a.discrepancy[ok]);bb=abs(b.discrepancy[ok]);assert ok.sum()==r.common_people and (aa<bb).sum()==r.improved_people;close(aa.mean(),r.straight_discrepancy);close(bb.mean(),r.turn_discrepancy);close((aa-bb).mean(),r.difference_straight_minus_turn)
        for _,r in views.iterrows():
            g=d[d.condition.eq(r.condition)&d.method.eq(r.method)];a=g[g.camera.eq('PG1')].set_index('subject');b=g[g.camera.eq('PG2')].set_index('subject');raw=a.raw_available&b.raw_available;ok=a.both_quality&b.both_quality;assert raw.sum()==r.raw_common_people and ok.sum()==r.both_quality_people;diff=abs(a.cadence-b.cadence);close(diff[raw].mean(),r.raw_view_difference);close(diff[ok].mean(),r.quality_view_difference)
    save(HERE/f'{mode}-verification.json',dict(passed=True,tasks=len(people),people=len(people),event_replays=2*len(people),period_replays=replays,new_model_fits=0,synthetic100spm_check=True,event_indices_exact=True,numeric_tolerance=1e-10,r2_tolerance=1e-9,verify_sha256=sha(__file__),study_sha256=sha(HERE/'study.py'),analysis_sha256=sha(HERE/'analyze.py')));print('VERIFIED',mode,len(people),replays,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
