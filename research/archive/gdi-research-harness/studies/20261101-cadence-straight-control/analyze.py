from study import *
GROUPS=[('RGB2D','PG1'),('RGB2D','PG2'),('RGB3D','PG1'),('RGB3D','PG2'),('Reference3D','PG1')]
def tables(periods,events):
    d=periods.merge(events[['subject','condition','cadence','quality','straight']],on=['subject','condition'],suffixes=('','_event'),validate='many_to_one');d=d[[tuple(x) in GROUPS for x in d[['method','camera']].values]].copy();d['discrepancy']=d.cadence-d.cadence_event;d['raw_available']=d.quality_event&d.discrepancy.notna();d['both_quality']=d.raw_available&d.quality
    metrics=[];geometry=[];views=[]
    for (condition,method,cam),g in d.groupby(['condition','method','camera'],sort=False):
        metrics.append(dict(condition=condition,method=method,camera=cam,people=len(g),event_quality=int(g.quality_event.sum()),period_quality=int(g.quality.sum()),raw_available=int(g.raw_available.sum()),both_quality=int(g.both_quality.sum()),actual_straight=int(g.straight.sum()),raw_discrepancy=abs(g.loc[g.raw_available,'discrepancy']).mean(),quality_discrepancy=abs(g.loc[g.both_quality,'discrepancy']).mean(),raw_bias=g.loc[g.raw_available,'discrepancy'].mean()))
    for method,cam in GROUPS:
        g=d[d.method.eq(method)&d.camera.eq(cam)];a=g[g.condition.eq('straight')].set_index('subject');b=g[g.condition.eq('turn')].set_index('subject');ok=a.raw_available&b.raw_available
        geometry.append(dict(method=method,camera=cam,common_people=int(ok.sum()),straight_discrepancy=abs(a.loc[ok,'discrepancy']).mean(),turn_discrepancy=abs(b.loc[ok,'discrepancy']).mean(),difference_straight_minus_turn=(abs(a.discrepancy)-abs(b.discrepancy))[ok].mean(),improved_people=int((abs(a.discrepancy[ok])<abs(b.discrepancy[ok])).sum())))
    for condition in CONDITIONS:
        for method in ['RGB2D','RGB3D']:
            g=d[d.condition.eq(condition)&d.method.eq(method)];a=g[g.camera.eq('PG1')].set_index('subject');b=g[g.camera.eq('PG2')].set_index('subject');raw=a.raw_available&b.raw_available;ok=a.both_quality&b.both_quality;diff=abs(a.cadence-b.cadence)
            views.append(dict(condition=condition,method=method,raw_common_people=int(raw.sum()),both_quality_people=int(ok.sum()),raw_view_difference=diff[raw].mean(),quality_view_difference=diff[ok].mean()))
    return d,pd.DataFrame(metrics),pd.DataFrame(geometry),pd.DataFrame(views)
if __name__=='__main__':
    d,m,g,v=tables(pd.read_csv(HERE/'main-estimates.csv'),pd.read_csv(HERE/'main-events.csv'))
    for n,t in [('comparisons',d),('metrics',m),('geometry-paired',g),('view-paired',v)]:t.to_csv(HERE/f'{n}.csv',index=False)
    save(HERE/'analysis-completion.json',dict(detail_rows=len(d),metric_rows=len(m),geometry_pairs=len(g),view_pairs=len(v),code_sha256=sha(__file__)))
