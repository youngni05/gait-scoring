"""Replay saved inputs and exercise the actual local upload/score path."""
import argparse
import json
import time
import urllib.request
import urllib.error
import engine

HERE=engine.HERE

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url', default='http://PRIVATE_HOST:8766')
    args=parser.parse_args()
    report={'checks':[], 'clinicalValidation':False}
    engine.check_assets()
    report['checks'].append('54 model hashes and MediaPipe hash match existing manifest')
    report['replay']=[]
    for sample in engine.MANIFEST['samples']:
        x=engine.np.fromfile(engine.ASSETS/sample['path'], dtype=engine.np.float32).reshape(sample['shape'])
        result=engine.predict(x)
        error=float(engine.np.max(engine.np.abs(engine.np.array([result['left'],result['right']])-sample['expected'])))
        assert error<=engine.MANIFEST['tolerance'], error
        report['replay'].append({'sample':sample['id'],'maxDifference':error})
    # Independently compare this entry point with the saved MediaPipe Full features.
    old=engine.ROOT/'gdi-research-harness/studies/20261101-self-video-mediapipe'
    expected=json.loads((old/'results.json').read_text(encoding='utf-8'))['summary']['primary']['best']
    raw=engine.np.load('D:/gait-research-data/D_P/self-video-mediapipe-20261101/clip1-pose.npz')['raw']
    x=engine.pre.extract_windows(raw)['x']
    result=engine.predict(x)
    diff=float(engine.np.max(engine.np.abs(engine.np.array([result['left'],result['right']])-expected['clip1']['left_right'])))
    assert diff<=1e-4, diff
    report['mediapipeSavedReplayMaxDifference']=diff
    print('Saved feature replay passed',flush=True)
    def request(path, body=None, token=None):
        headers={'X-Demo-Token':token} if token else {}
        with urllib.request.urlopen(urllib.request.Request(args.url+path,data=body,headers=headers),timeout=90) as r:
            data=r.read()
            return r.status, json.loads(data) if 'application/json' in r.headers.get('Content-Type','') else data
    assert request('/')[0]==200
    token=request('/api/config')[1]['token']
    try:
        request('/api/score',b'')
        raise AssertionError('Missing token accepted')
    except urllib.error.HTTPError as e: assert e.code==403
    try:
        request('/api/score',b'',token)
        raise AssertionError('Score before pose accepted')
    except urllib.error.HTTPError as e: assert e.code==409
    assert request('/api/upload',b'not a video',token)[0]==202
    def wait(stage):
        end=time.monotonic()+240
        while time.monotonic()<end:
            job=request('/api/job',token=token)[1]
            if job['stage'] in (stage,'error'):
                return job
            time.sleep(.7)
        raise AssertionError('Timed out')
    assert wait('error')['stage']=='error'
    for name,frames in [('short',60),('no-person',125)]:
        path=engine.Path('D:/gait-research-data/D_P/meeting-demo-20261106/uploads')/(name+'-test.mp4')
        try:
            writer=engine.cv2.VideoWriter(str(path),engine.cv2.VideoWriter_fourcc(*'mp4v'),30,(320,240))
            assert writer.isOpened()
            for _ in range(frames):writer.write(engine.np.zeros((240,320,3),dtype=engine.np.uint8))
            writer.release()
            request('/api/upload',path.read_bytes(),token)
            job=wait('error')
            assert job['stage']=='error' and 'result' not in job
            report['checks'].append(name+' input rejected without a score')
        finally:path.unlink(missing_ok=True)
    print('HTTP and invalid-input checks passed. Uploading real normal video.',flush=True)
    normal=engine.ROOT/'영상자료/검증용/정상 보행.mp4'
    request('/api/upload',normal.read_bytes(),token)
    job=wait('review')
    assert job['stage']=='review',job.get('message')
    assert job['review']['windows']==13 and job['review']['frames']==1353
    request('/api/score',b'',token)
    job=wait('complete')
    assert job['stage']=='complete',job.get('message')
    result=job['result']
    rows=result['comparison']['rows']
    assert len(rows)==6 and len({r['id'] for r in rows})==6
    assert all(engine.np.isfinite([r['left'],r['right'],r['mean']]).all() for r in rows)
    fusion=next(r for r in rows if r['id']=='baseline_fusion')
    assert abs(fusion['left']-result['left'])<1e-10 and abs(fusion['right']-result['right'])<1e-10
    report['sixModelComparison']=rows
    diff=float(engine.np.max(engine.np.abs(engine.np.array([result['left'],result['right']])-expected['clip1']['left_right'])))
    assert diff<=1e-4,diff
    report['uploadedVideo']={'source':'existing user normal-walk clip; no actual GDI label','frames':1353,'windows':13,'left':result['left'],'right':result['right'],'maxDifferenceFromPriorMediaPipe':diff}
    # Wait for the worker's finally clause before checking temporary-file cleanup.
    time.sleep(.2)
    assert not list(engine.Path('D:/gait-research-data/D_P/meeting-demo-20261106/uploads').glob('*.mp4'))
    report['checks'].extend(['real video upload -> MediaPipe -> same features -> original54 and adult64-added55 checkpoints -> six results', 'temporary videos removed', 'HTTP page and request protection passed'])
    report['passed']=True
    (HERE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
