from study import *

def run():
    d=pd.read_csv(HERE/'main-clips.csv');assert len(d)==56 and d.subject.nunique()==14 and not d.duplicated(['subject','camera','condition']).any()
    rows=[];pairs=[]
    for condition in ['straight','turn']:
        for method in ['plane_net','plane_ols','pinhole_ols']:
            for cam in ['PG1','PG2']:
                g=d[d.condition.eq(condition)&d.camera.eq(cam)];truth=g.truth_net if method=='plane_net' else g.truth_ols;e=g[method]-truth
                rows.append(dict(condition=condition,camera=cam,method=method,people=len(g),mae=float(abs(e).mean()),bias=float(e.mean()),median_error=float(abs(e).median()),p90_error=float(abs(e).quantile(.9)),mean_truth=float(truth.mean()),mean_estimated=float(g[method].mean()),height_2cm_mean_change=float((g[method]*.02/g.height).mean())))
            wide=d[d.condition.eq(condition)].pivot(index='subject',columns='camera',values=method);diff=wide.PG2-wide.PG1
            pairs.append(dict(condition=condition,method=method,people=len(wide),absolute_view_difference=float(abs(diff).mean()),signed_view_difference=float(diff.mean()),p90_absolute_view_difference=float(abs(diff).quantile(.9))))
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False);pd.DataFrame(pairs).to_csv(HERE/'paired.csv',index=False)
    save(HERE/'analysis.json',dict(metric_rows=len(rows),paired_rows=len(pairs),analysis_sha256=sha(__file__)))
if __name__=='__main__':run()
