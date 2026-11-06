from study import *

def run(mode):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for q,h in protocol['sources'].items():assert sha(q)==h
    rows=clips(mode);worst=0.
    for c in rows:
        q=json.loads((STORE/f"{c['key']}.json").read_text());assert q['code_sha256']==sha(HERE/'study.py') and sha(q['array_path'])==q['array_sha256']
        for path,digest in q['sources'].items():assert sha(path)==digest
        z=np.load(q['array_path']);source=np.load(next(path for path in q['sources'] if path.endswith(f"{c['camera']}_800.npz")))
        np.testing.assert_array_equal(z['frames'],np.arange(c['frames_one_based'][0],c['frames_one_based'][1]+1))
        with tarfile.open(SOURCE/'raw/Camera Parameters.tar') as tar:
            original_k=np.load(io.BytesIO(tar.extractfile(f"Calib/cameraParams_{c['camera']}.npz").read()))['IntrinsicMatrix']
        np.testing.assert_array_equal(z['intrinsic'],original_k)
        reference=np.load(next(path for path in q['sources'] if path.endswith('_comparison.npz')))
        a=np.searchsorted(source['video_frame_one_based'],z['frames']);b=np.searchsorted(reference['frames'],z['frames'])
        np.testing.assert_array_equal(z['lm'],source['landmarks'][a]);np.testing.assert_array_equal(z['reference'],reference['reference_3d'][b])
        meta=next(path for path in q['sources'] if path.endswith('.mat'));h=loadmat(meta,simplify_cells=True)[f"Subject_{c['subject']}_F"]['subject']['height']/100.
        assert float(z['height'])==h==q['height']
        lm=z['lm'];pix=lm[:,:,:2]*np.array([800.,600.]);hip=(pix[:,23]+pix[:,24])/2.;span=pix[:,:,1].max(1)-pix[:,:,1].min(1);scale=h/np.median(span)
        t=np.arange(60,dtype=float)/30.;design=np.c_[np.ones(60),t]
        trend=lambda x:np.linalg.lstsq(design,x,rcond=None)[0][1]
        actual=(z['reference'][:,6]+z['reference'][:,7])/2000.
        k=z['intrinsic'];depth=k[1,1]*h/span;xyz=np.c_[(hip[:,0]-k[2,0])*depth/k[0,0],(hip[:,1]-k[2,1])*depth/k[1,1],depth]
        computed={'plane_net':np.linalg.norm(hip[-1]-hip[0])*scale/(59/30.),'plane_ols':np.linalg.norm(trend(hip))*scale,'pinhole_ols':np.linalg.norm(trend(xyz)),'truth_net':np.linalg.norm(actual[-1,:2]-actual[0,:2])/(59/30.),'truth_ols':np.linalg.norm(trend(actual[:,:2]))}
        for key,value in computed.items():error=abs(value-q[key]);assert error<1e-10,(c['key'],key,error);worst=max(worst,error)
        # With height the sole metric scale, +/-2cm changes all estimates linearly.
        for offset in [-.02,.02]:
            result,_=measure(lm,z['reference'],h+offset,k)
            for method in ['plane_net','plane_ols','pinhole_ols']:assert abs(result[method]-q[method]*(h+offset)/h)<1e-10
    for sid in {c['subject'] for c in rows}:
        for condition in ['straight','turn']:
            a=np.load(STORE/f's{sid:02d}_PG1_{condition}.npz');b=np.load(STORE/f's{sid:02d}_PG2_{condition}.npz')
            np.testing.assert_array_equal(a['frames'],b['frames']);np.testing.assert_array_equal(a['reference'],b['reference'])
    table=pd.read_csv(HERE/f'{mode}-clips.csv');assert len(table)==len(rows) and not table.key.duplicated().any()
    for _,r in table.iterrows():
        q=json.loads((STORE/f'{r.key}.json').read_text())
        for key in ['plane_net','plane_ols','pinhole_ols','truth_net','truth_ols','height']:
            assert abs(r[key]-q[key])<1e-12
    if mode=='main':
        metrics=pd.read_csv(HERE/'metrics.csv');pairs=pd.read_csv(HERE/'paired.csv');d=pd.read_csv(HERE/'main-clips.csv');assert len(metrics)==12 and len(pairs)==6
        for _,r in metrics.iterrows():
            g=d[d.condition.eq(r.condition)&d.camera.eq(r.camera)];truth=g.truth_net if r.method=='plane_net' else g.truth_ols;e=g[r.method]-truth
            assert abs(abs(e).mean()-r.mae)<1e-10 and abs(e.mean()-r.bias)<1e-10
        for _,r in pairs.iterrows():
            g=d[d.condition.eq(r.condition)].pivot(index='subject',columns='camera',values=r.method);diff=g.PG2-g.PG1
            assert abs(abs(diff).mean()-r.absolute_view_difference)<1e-10 and abs(diff.mean()-r.signed_view_difference)<1e-10
    save(HERE/f'{mode}-verification.json',dict(passed=True,clips=len(rows),people=len(set(c['subject'] for c in rows)),max_independent_error=worst,tolerance=1e-10,verify_sha256=sha(__file__),study_sha256=sha(HERE/'study.py')));print('VERIFIED',mode,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
