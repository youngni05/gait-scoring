from study import *
def run():
    data=source();d=pd.concat([pd.read_csv(STORE/f's{s:02d}-train{c}.csv') for s,c in tasks('main',data)],ignore_index=True)
    assert len(d)==112 and not d.duplicated(['key','training_camera']).any()
    rows=[];camera_rows=[];pairs=[]
    for condition in ['straight','turn']:
        for transfer in ['same_camera','opposite_camera']:
            g=d[d.condition.eq(condition)&d.transfer.eq(transfer)]
            for method in ['RawPinhole','ScaledPinhole']:
                e=g[method]-g.truth_ols;person=pd.DataFrame(dict(subject=g.subject,error=e,ae=abs(e))).groupby('subject').mean()
                rows.append(dict(condition=condition,transfer=transfer,method=method,people=len(person),mae=float(person.ae.mean()),bias=float(person.error.mean()),p90_person_error=float(person.ae.quantile(.9))))
                for cam in ['PG1','PG2']:
                    sub=g[g.camera.eq(cam)];z=sub[method]-sub.truth_ols
                    camera_rows.append(dict(condition=condition,transfer=transfer,method=method,camera=cam,people=sub.subject.nunique(),mae=float(abs(z).mean()),bias=float(z.mean())))
        for train_cam in ['PG1','PG2']:
            g=d[d.condition.eq(condition)&d.training_camera.eq(train_cam)]
            for method in ['RawPinhole','ScaledPinhole']:
                wide=g.pivot(index='subject',columns='camera',values=method);diff=wide.PG2-wide.PG1
                pairs.append(dict(condition=condition,training_camera=train_cam,method=method,people=len(wide),absolute_view_difference=float(abs(diff).mean()),signed_view_difference=float(diff.mean())))
    d.to_csv(STORE/'predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False);pd.DataFrame(camera_rows).to_csv(HERE/'camera-metrics.csv',index=False);pd.DataFrame(pairs).to_csv(HERE/'paired.csv',index=False)
    save(HERE/'analysis.json',dict(predictions=112,metric_rows=8,camera_metric_rows=16,paired_rows=8,analysis_sha256=sha(__file__),predictions_sha256=sha(STORE/'predictions.csv')))
if __name__=='__main__':run()
