"""Full raw replay and algebra/unit/source checks; no threshold changes."""
from pathlib import Path
import json,hashlib,subprocess,sys,csv
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads((HERE/'harmonization-protocol.json').read_text());assert sha(HERE/'harmonize.py')==p['code_sha256']
    for path,h in p['sources'].items():assert sha(path)==h
    source=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');acq=json.loads((source/'acquisition.json').read_text());raw=next(r for r in acq if r['name']=='IMU_signals.zip');assert sha(raw['path'])==raw['sha256']
    expected=sha(BIG/'F-features.json');old=(BIG/'F-features.json').read_bytes()
    subprocess.run([str(source/'.venv/Scripts/python.exe'),'-u',str(HERE/'harmonize.py'),'F'],check=True)
    assert (BIG/'F-features.json').read_bytes()==old and sha(BIG/'F-features.json')==expected
    rows=json.loads(old);ok=[r for r in rows if r['status']=='ok'];assert len(rows)==594 and len(ok)==229
    for r in ok:
        a,b=r['events'];sp,dt,sw=r['X'];assert a['end']==b['start']
        assert abs(dt-(b['ic']-a['ic'])/100)<1e-12 and abs(sw-(b['ic']-b['tc'])/(b['ic']-a['ic']))<1e-12
        assert abs(sp*dt-r['stride_length'])<1e-12 and b['end']-a['start']<=400
        assert all(x['samples']==x['expected_samples'] for x in r['checks'])
    g=json.loads((BIG/'G-features.json').read_text());assert len(g)==142
    root=Path('D:/gait-research-data/D_P/planar-tug-improvement-20261104/gstride/processed/_GAIT')
    for r in g:
        path=root/'_PARAMETERS'/f'{r["id"]}_GAIT_PARAMETERS.csv';source_row=next(q for q in csv.DictReader(path.open()) if int(float(q['index']))==r['index'])
        speed,dt,sw=r['X'];assert abs(float(source_row['CycleDuration[s]'])-dt)<1e-12 and abs(float(source_row['Per_Swing[%CD]'])/100-sw)<1e-12 and abs(speed*dt-r['stride_length'])<1e-7
    result={'status':'passed','raw_source_sha256':raw['sha256'],'F_full_raw_replay_people':594,'F_identical_measurable':229,'F_identical_unmeasurable':365,'G_formula_rebuilds':142,'exact_F_feature_bytes_identical':True,
      'different_event_detectors_remain':True,'clinical_gait_parameter_accuracy_validated':False,'features_sha256':{'F':expected,'G':sha(BIG/'G-features.json')}}
    (HERE/'feature-verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('FEATURE_REPLAY_VERIFIED 594,229,142',flush=True)
if __name__=='__main__':main()
