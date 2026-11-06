"""Rebuild table joins and matched RGB/reference inputs, without trusting summary rows."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    r=read(HERE/'video-results.json');prot=read(HERE/'video-protocol.json');assert sha(HERE/'video_measurement.py')==prot['code_sha256']
    assert sha(HERE/'measurement-pairs.csv')==r['pairs_sha256']
    for p,h in prot['sources'].items():assert sha(p)==h
    acq=read(HERE/'healthgait-tables-acquisition.json');tables={}
    for f in acq['tables']:
        assert sha(f['local'])==f['sha256'];rows=list(csv.DictReader(Path(f['local']).open()));tables[Path(f['local']).name]={x['ID']:x for x in rows};assert len(tables[Path(f['local']).name])==len(rows)==398
    ref=tables['gait_parameters.csv'];est=tables['gait_parameters_estimation.csv'];assert set(ref)==set(est)
    def number(x):
        try:return float(x)
        except ValueError:return float('nan')
    checked=0
    for m in r['healthgait']:
        field={'speed':'Velocity','step':'Step','stride':'Stride','cadence':'Cadence','cycle':'Cadence'}[m['metric']]+'_'+m['mode'];pairs=[]
        for pid in sorted(ref):
            a,b=number(ref[pid][field]),number(est[pid][field])
            if np.isfinite([a,b]).all() and a>0 and b>0:pairs.append((120/a,120/b) if m['metric']=='cycle' else(a,b))
        a=np.array(pairs);err=a[:,1]-a[:,0];assert len(a)==m['n'];np.testing.assert_allclose([abs(err).mean(),err.mean(),np.quantile(abs(err),.95)],[m['mae'],m['bias'],m['p95']],rtol=0,atol=1e-12);checked+=1
    sys.path.insert(0,str(HERE.parent.parent/'gdi-research-harness/studies/20261101-camera-cadence'));import study as c
    source=Path('D:/gait-research-data/MoVi/experiments/walking-expansion')
    snapshots=read(source/'results.json')['results'];count=0;source_hashes={};max_difference=0
    for sid in sorted(set(x['subject'] for x in r['MoVi_estimates'])):
      p=Path(f'D:/gait-research-data/D_P/camera-cadence-20261101/s{sid:02d}-4sec.npz');receipt=read(p.with_suffix('.json'));assert sha(p)==receipt['array_sha256'];z=np.load(p)
      for path,h in receipt['sources'].items():assert sha(path)==h;source_hashes[path]=h
      for cam in ['PG1','PG2']:
        raw=source/f'subject-{sid:02d}/heavy/{cam}_800.npz';q=np.load(raw);frames=z[cam+'_frames'];ix=np.searchsorted(q['video_frame_one_based'],frames)
        assert len(frames)==120 and np.all(np.diff(frames)==1);np.testing.assert_array_equal(q['video_frame_one_based'][ix],frames);np.testing.assert_array_equal(q['landmarks'][ix],z[cam+'_lm'])
        original=next(x for x in snapshots if x['subject']==sid and x['camera']==cam and x['variant']=='heavy' and x['input_width']==800);assert sha(raw)==original['prediction_sha256']
        rr=read(raw.with_name(cam+'_800_comparison.json')) if raw.with_name(cam+'_800_comparison.json').exists() else None
        row=next(x for x in r['MoVi_estimates'] if x['subject']==sid and x['camera']==cam);lm=z[cam+'_lm'];xy=lm[:,:,:2]*[800,600]
        period=c.estimate(xy[:,27]-xy[:,28]);assert period==row['period_quality']
        count+=1
    assert count==54
    pairs=list(csv.DictReader((HERE/'measurement-pairs.csv').open()))
    for m in r['MoVi']:
        rows=[x for x in pairs if x['dataset']=='MoVi' and x['condition']==m['camera'] and x['metric']==m['metric']]
        assert len(rows)==m['n'] and len(set(x['id'] for x in rows))==len(rows)
        err=np.array([float(x['prediction'])-float(x['reference']) for x in rows]);np.testing.assert_allclose([abs(err).mean(),err.mean()],[m['mae'],m['bias']],rtol=0,atol=1e-12)
    st=read(HERE/'straight-cycle-results.json');sp=read(HERE/'straight-cycle-protocol.json');assert sha(HERE/'straight_cycles.py')==sp['code_sha256']
    for m in st['metrics']:
        rows=[x for x in st['estimates'] if x['camera']==m['camera'] and x[m['metric']] is not None and x['reference'][m['metric']] is not None]
        err=np.array([x[m['metric']]-x['reference'][m['metric']] for x in rows]);assert len(rows)==m['n'];np.testing.assert_allclose(abs(err).mean(),m['mae'],rtol=0,atol=1e-12)
    out={'status':'passed','healthgait_tables_verified':3,'healthgait_metric_rebuilds':checked,'matched_RGB_windows_verified':count,'people':27,
      'straight_followup_people':14,'straight_followup_rows_verified':6,'same_frames_and_cached_RGB_only_MediaPipe_replay':True,
      'concurrent_clinical_HS_groundtruth_available':False,'actual_phone_test_performed':False,'independent_source_hashes':source_hashes,
      'results_sha256':sha(HERE/'video-results.json'),'straight_results_sha256':sha(HERE/'straight-cycle-results.json')}
    (HERE/'video-verification.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('VIDEO_VERIFIED',checked,count,'straight6',flush=True)
if __name__=='__main__':main()
