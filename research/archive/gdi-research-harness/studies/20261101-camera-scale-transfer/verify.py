from study import *
def run(mode):
    d=source();ts=tasks(mode,d);worst=0.
    for sid,cam in ts:
        tag=f's{sid:02d}-train{cam}';q=json.loads((STORE/f'{tag}.json').read_text());model=json.loads(Path(q['model_path']).read_text());assert sha(q['model_path'])==q['model_sha256'] and sha(q['prediction_path'])==q['prediction_sha256'] and q['code_sha256']==sha(HERE/'study.py')
        a=d[d.subject.ne(sid)&d.camera.eq(cam)&d.condition.eq('straight')].sort_values('subject');assert a.subject.tolist()==q['training_people']==model['training_people'] and len(a)==13 and sid not in a.subject.tolist();assert a.key.tolist()==q['training_keys']
        alpha=float(np.linalg.lstsq(a[['pinhole_ols']].to_numpy(),a.truth_ols.to_numpy(),rcond=None)[0][0]);error=abs(alpha-model['alpha']);assert error<1e-12;worst=max(worst,error)
        pred=pd.read_csv(q['prediction_path']);test=d[d.subject.eq(sid)].sort_values(['condition','camera']);np.testing.assert_array_equal(pred.key,test.key)
        np.testing.assert_allclose(pred.truth_ols,test.truth_ols,rtol=0,atol=1e-12);np.testing.assert_allclose(pred.RawPinhole,test.pinhole_ols,rtol=0,atol=1e-12)
        np.testing.assert_allclose(pred.ScaledPinhole,test.pinhole_ols.to_numpy()*alpha,rtol=0,atol=1e-12);assert (pred.subject==sid).all() and (pred.training_camera==cam).all()
        np.testing.assert_array_equal(pred.transfer,np.where(pred.camera.eq(cam),'same_camera','opposite_camera'))
    if mode=='main':
        pred=pd.read_csv(STORE/'predictions.csv');assert len(pred)==112 and pred.subject.nunique()==14
        for sid,cam in ts:
            expected=pd.read_csv(STORE/f's{sid:02d}-train{cam}.csv');actual=pred[pred.subject.eq(sid)&pred.training_camera.eq(cam)].reset_index(drop=True);pd.testing.assert_frame_equal(actual,expected)
        m=pd.read_csv(HERE/'metrics.csv');cm=pd.read_csv(HERE/'camera-metrics.csv');p=pd.read_csv(HERE/'paired.csv');assert (len(m),len(cm),len(p))==(8,16,8)
        for frame in [m,cm]:
            for _,r in frame.iterrows():
                g=pred[pred.condition.eq(r.condition)&pred.transfer.eq(r.transfer)]
                if 'camera' in frame.columns:g=g[g.camera.eq(r.camera)]
                e=g[r.method]-g.truth_ols;person=pd.DataFrame(dict(subject=g.subject,error=e,ae=abs(e))).groupby('subject').mean()
                assert abs(person.ae.mean()-r.mae)<1e-12 and abs(person.error.mean()-r.bias)<1e-12
        for _,r in p.iterrows():
            g=pred[pred.condition.eq(r.condition)&pred.training_camera.eq(r.training_camera)];a=g[g.camera.eq('PG1')].set_index('subject')[r.method];b=g[g.camera.eq('PG2')].set_index('subject')[r.method];diff=b-a
            assert abs(abs(diff).mean()-r.absolute_view_difference)<1e-12 and abs(diff.mean()-r.signed_view_difference)<1e-12
    save(HERE/f'{mode}-verification.json',dict(passed=True,tasks=len(ts),scalar_replays=len(ts),independent_refits=len(ts),predictions=4*len(ts),people=len(ts)//2,max_scalar_difference=worst,tolerance=1e-12,verify_sha256=sha(__file__),study_sha256=sha(HERE/'study.py')));print('VERIFIED',mode,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
