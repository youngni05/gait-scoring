from analyze import *
from scipy.ndimage import uniform_filter1d

def main():
    sys.path.insert(0,str(CHAIR))
    spec=importlib.util.spec_from_file_location('audit_segment',CHAIR/'segment.py');segment=importlib.util.module_from_spec(spec);spec.loader.exec_module(segment)
    rows=json.loads((HERE/'person-audit.json').read_text());manifest={m['id']:m for m in json.loads((PRIOR/'include283/manifest.json').read_text())};gaps=pd.read_csv(HERE/'gap-review.csv');count=0
    for r in rows:
        m=manifest[r['id']];a=read(source.long.TURN/'raw'/r['source_file']);assert sha(source.long.TURN/'raw'/r['source_file'])==r['source_sha256']
        for kind in ['rise','sit']:
            try:actual=segment.transition(a,m,kind)
            except ValueError as e:actual={'error':str(e)}
            assert actual==r['phases'][kind];count+=1
        for g in gaps[gaps.id==r['id']].itertuples():
            t=a['time'];sel=(t>=g.start)&(t<=g.end);tt=t[sel];pos=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);smooth=uniform_filter1d(pos,max(3,round(.25/np.median(np.diff(t)))),axis=0,mode='nearest');sp=np.linalg.norm(np.gradient(smooth,t,axis=0),axis=1)
            horizontal=np.linalg.norm(np.diff(pos[sel],axis=0),axis=1)/np.diff(tt);vertical=abs(np.diff(a['pelvis_ty'][sel])/np.diff(tt))
            assert np.isclose(max(horizontal),g.native_horizontal_speed_max) and np.isclose(max(vertical),g.native_vertical_speed_max)
            assert abs(g.seconds-g.inside_observation20-g.outside_observation20)<1e-10
            assert abs(g.smoothed_speed_median-np.median(sp[sel]))<1e-10
            mids=(tt[:-1]+tt[1:])/2;bad=np.zeros(len(mids),bool)
            for j in np.flatnonzero((horizontal>6)|(vertical>3)):bad|=(mids>=tt[j]-.15)&(mids<=tt[j+1]+.15)
            assert abs(np.diff(tt)[bad].sum()-g.jump_masked_seconds)<1e-7
            assert abs(np.diff(tt)[(~bad)&(sp[sel][:-1]<.15)].sum()-g.clean_below015_seconds)<1e-7
    labels=pd.read_csv(LABELS);valid=labels[labels.tft_tug>0];tail=valid[valid.tft_tug>=20];cov=json.loads((HERE/'clinical-coverage.json').read_text())
    assert [len(labels),labels.subid.nunique(),len(valid),valid.subid.nunique(),len(tail),tail.subid.nunique()]==[cov[k] for k in ['rows','unique_people','positive_tug_rows','positive_tug_people','long_visits','long_people']]
    large=[r for r in rows if r['large_under']];assert len(large)==4 and all(r['gait_stop20']==0 and r['tracking_removed20']==0 for r in large)
    save('audit-verification.json',{'passed':True,'phase_results_recomputed_from_raw':count,'gap_segments_recomputed':len(gaps),'positive_tug_people':valid.subid.nunique(),'long_people':tail.subid.nunique(),'large_under_people':len(large),'label_values_and_id_visit_matches':True})
    print('AUDIT_VERIFIED',count,'phase results',len(gaps),'gap segments',flush=True)
if __name__=='__main__':main()
