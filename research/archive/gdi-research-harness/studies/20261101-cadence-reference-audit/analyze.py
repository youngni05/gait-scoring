from study import *
def run():
    events=pd.read_csv(HERE/'main-events.csv');assert len(events)==27 and events.subject.nunique()==27;old=pd.read_csv(REFERENCE/'main-estimates.csv');old=old[old.seconds.eq(4)];rows=[]
    for _,e in events.iterrows():
        for method,camera in [('Reference3D','PG1'),('RGB2D','PG1'),('RGB2D','PG2'),('RGB3D','PG1'),('RGB3D','PG2')]:
            r=old[old.subject.eq(e.subject)&old.method.eq(method)&old.camera.eq(camera)].iloc[0];good=bool(e.quality and r.quality and pd.notna(e.cadence) and pd.notna(r.cadence));error=float(r.cadence-e.cadence) if pd.notna(r.cadence) and pd.notna(e.cadence) else np.nan
            half=bool(good and abs(r.cadence-e.cadence/2)<=5 and abs(error)>5)
            rows.append(dict(subject=int(e.subject),method=method,camera=camera,cadence=r.cadence,period_quality=r.quality,event_cadence=e.cadence,event_quality=e.quality,straight=e.straight,both_quality=good,discrepancy=error,half_proxy_flag=half))
    detail=pd.DataFrame(rows);metrics=[]
    for (method,camera),g in detail.groupby(['method','camera']):
        both=g.both_quality;straight=both&g.straight
        metrics.append(dict(method=method,camera=camera,selected_people=len(g),event_quality_people=int(g.event_quality.sum()),period_quality_people=int(g.period_quality.sum()),both_quality_people=int(both.sum()),mean_absolute_discrepancy=float(abs(g.discrepancy[both]).mean()),bias=float(g.discrepancy[both].mean()),half_proxy_people=int(g.half_proxy_flag.sum()),straight_both_people=int(straight.sum()),straight_mean_absolute_discrepancy=float(abs(g.discrepancy[straight]).mean())))
    pairs=[]
    for method in ['RGB2D','RGB3D']:
        a=detail[detail.method.eq(method)&detail.camera.eq('PG1')].set_index('subject');b=detail[detail.method.eq(method)&detail.camera.eq('PG2')].set_index('subject');ok=a.both_quality&b.both_quality;diff=b.cadence-a.cadence
        pairs.append(dict(method=method,selected_people=27,both_camera_event_quality_people=int(ok.sum()),mean_absolute_view_difference=float(abs(diff[ok]).mean())))
    detail.to_csv(HERE/'comparisons.csv',index=False);pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False);pd.DataFrame(pairs).to_csv(HERE/'paired.csv',index=False);save(HERE/'analysis.json',dict(people=27,detail_rows=len(detail),metric_rows=len(metrics),paired_rows=len(pairs),analysis_sha256=sha(__file__)))
if __name__=='__main__':run()
