from study import *
def tables(d):
    metrics=[];paired=[];proxies=[]
    for cam in ['PG1','PG2']:
        a=d[d.camera.eq(cam)&d.condition.eq('straight')];b=d[d.camera.eq(cam)&d.condition.eq('turn')];metrics.append(dict(camera=cam,straight_people=len(a),straight_accepted=int(a.accepted.sum()),straight_retention=a.accepted.mean(),turn_people=len(b),turn_accepted=int(b.accepted.sum()),turn_false_acceptance=b.accepted.mean()))
    for condition in source.CONDITIONS:
        g=d[d.condition.eq(condition)];a=g[g.camera.eq('PG1')].set_index('subject');b=g[g.camera.eq('PG2')].set_index('subject');paired.append(dict(condition=condition,people=len(a),both_accepted=int((a.accepted&b.accepted).sum()),one_view_disagrees=int((a.accepted!=b.accepted).sum()),both_rejected=int((~a.accepted&~b.accepted).sum())))
    old=pd.read_csv(source.HERE/'comparisons.csv');old=old[old.method.isin(['RGB2D','RGB3D'])];merged=old.merge(d[['subject','condition','camera','accepted']],on=['subject','condition','camera'],validate='many_to_one')
    for (condition,method,cam),g in merged.groupby(['condition','method','camera'],sort=False):
        ok=g.accepted&g.raw_available;quality=ok&g.quality;proxies.append(dict(condition=condition,method=method,camera=cam,people=len(g),gate_accepted=int(g.accepted.sum()),event_available=int(g.raw_available.sum()),gated_event_available=int(ok.sum()),gated_period_quality=int(quality.sum()),ungated_discrepancy=abs(g.discrepancy[g.raw_available]).mean(),gated_discrepancy=abs(g.discrepancy[ok]).mean()))
    return pd.DataFrame(metrics),pd.DataFrame(paired),pd.DataFrame(proxies)
if __name__=='__main__':
    for n,t in zip(['metrics','paired','proxy-comparison'],tables(pd.read_csv(HERE/'main-gates.csv'))):t.to_csv(HERE/f'{n}.csv',index=False)
    save(HERE/'analysis-completion.json',dict(metric_rows=2,paired_rows=2,proxy_rows=8,code_sha256=sha(__file__)))
