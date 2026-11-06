"""Same-person, equal-time simple Ridge versus sitting information."""
from pathlib import Path
import sys,json,hashlib,importlib.util
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261031-opencap-leave283/prep'
EXT=HERE.parent/'20261031-opencap-extrapolation'
spec=importlib.util.spec_from_file_location('ext_study',EXT/'study.py')
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
np,pd=ext.np,ext.pd
import joblib
ALPHAS=[.1,1.,10.,100.]
VIEWS=['ShortSpeedTurn','MatchedSpeedTurn','PlusSitTime','PlusSitMotion','PlusSitAll']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def fit(x,y,tr,te,alpha):return ext.fit(x,y,tr,te,'Ridge',alpha)
def imports():
    sys.path.insert(0,str(HERE.parent/'20261031-opencap-turn'))
    import common,prepare,geometry
    return common,prepare,geometry

def prepare_inputs():
    assert not (HERE/'protocol.json').exists()
    common,prepare,geometry=imports()
    d=dict(np.load(PRIOR/'input.npz'));rows=json.loads((PRIOR/'manifest.json').read_text());full=d['Full']
    assert len(d['ids'])==100 and 'sub-283' not in d['ids'] and sum(d['y']>=20)==3
    matched=[];audit=[];source_hashes={}
    for name in ['input.npz','manifest.json','splits.json']:
        source_hashes[str((PRIOR/name).relative_to(HERE.parent))]=sha(PRIOR/name)
    for file in [EXT/'study.py',HERE.parent/'20261031-opencap-turn/prepare.py',HERE.parent/'20261030-opencap415-tug-timeseries/prepare.py',HERE.parent/'20261031-opencap-chair/segment.py']:
        source_hashes[str(file.relative_to(HERE.parent))]=sha(file)
    for i,m in enumerate(rows):
        assert m['id']==d['ids'][i] and m['target']==d['y'][i]
        sit=m['phase_windows']['sit'][1]-m['phase_windows']['sit'][0]
        duration=m['turn_observed_seconds'];start=m['walk_start'];end=start+3+sit
        assert abs(float(full[i,3])-duration)<1e-9 and abs(float(full[i,21])-sit)<1e-9
        assert end<=m['walk_end']+1e-7
        path=common.SOURCE/'raw'/m['walk_file'];assert sha(path)==m['walk_sha256']
        a=common.read(path);base,_=prepare.walk_window(a,start,3);extra,_=prepare.walk_window(a,start,3+sit)
        assert np.allclose(base,full[i,:3],atol=1e-12,rtol=0)
        native_max=geometry.root_qc(a,start,end);assert native_max<=6
        # Speed extraction returns float32; promote before appending the unchanged float64 duration.
        matched.append(np.concatenate([extra.astype(np.float64),full[i,3:4]]))
        audit.append({'id':m['id'],'target':m['target'],'walk_start':start,'control_walk_end':end,'original_clean_walk_end':m['walk_end'],
            'control_walk_seconds':3+sit,'added_walk_seconds':sit,'turn_seconds':duration,'sit_seconds':sit,
            'control_input_seconds':3+sit+duration,'sitting_input_seconds':3+duration+sit,
            'short_reference_input_seconds':3+duration,'previous_full_input_seconds':m['observed_seconds'],
            'max_control_walk_speed':native_max,'walk_sha256':m['walk_sha256']})
    z={'ShortSpeedTurn':full[:,:4], 'MatchedSpeedTurn':np.stack(matched),
       'PlusSitTime':full[:,[0,1,2,3,21]], 'PlusSitMotion':full[:,list(range(4))+list(range(22,29))],
       'PlusSitAll':full[:,list(range(4))+list(range(21,29))]}
    np.savez_compressed(HERE/'input.npz',ids=d['ids'],y=d['y'],**z)
    (HERE/'splits.json').write_bytes((PRIOR/'splits.json').read_bytes())
    (HERE/'manifest.json').write_bytes((PRIOR/'manifest.json').read_bytes())
    save('observation-audit.json',audit)
    protocol={'people':100,'long_people':3,'excluded':'sub-283 absent from all fitting, inner selection and evaluation',
        'target':'unchanged same-person same-visit clinical tft_tug seconds',
        'splits':'exact previous100-person outer5partitions x5fold and inner3fold; no reassignment',
        'model':'Ridge SVD, train-only StandardScaler, inner MAE alpha choice','alpha_grid':ALPHAS,
        'views':VIEWS,'primary_comparison':'PlusSitAll minus MatchedSpeedTurn; other views explanatory',
        'speed':'existing horizontal speed mean, standard deviation and mean absolute adjacent speed change; not only a single average speed',
        'control':'(3+S)s straight walking speed summaries + D-second turn duration',
        'sitting':'3s straight walking speed summaries + same D-second turn duration + S-second sitting features',
        'observation':'both 3+D+S seconds; all original clips and sit/turn boundaries preserved; control extends only inside previous QC-passed walking window; prior preparation/rise clips unused in both arms',
        'short_reference':'3+D seconds, informative-feature addition reference only, NOT equal-input-time comparison',
        'sit_features':['seconds','height_change','height_range','vertical_velocity_mean','vertical_velocity_std','vertical_speed_p95','vertical_speed_change','stationary_fraction'],
        'postprocessing':'negative predictions floored0, raw saved, no upper cap',
        'metrics':'person-equal MAE, positive underprediction, overprediction; >=20 descriptive, not clinical; conditional paired-person bootstrap10000 seed20261031; per-person and per-partition results',
        'limits':['development comparison after prior analyses, not independent validation','full record used to locate clips; input seconds not acquisition time','exact stopwatch cone/line trial and phase manual labels unresolved','sitting quality selection keeps previous100; not whole237 cohort'],
        'source_hashes':source_hashes,'input_hashes':{n:sha(HERE/n) for n in ['input.npz','manifest.json','splits.json','observation-audit.json']},'code_sha256':sha(HERE/'study.py')}
    save('protocol.json',protocol)
    print(json.dumps({'people':100,'long':3,'models':125,'matched_seconds':[min(r['control_input_seconds'] for r in audit),max(r['control_input_seconds'] for r in audit)]}),flush=True)

if __name__=='__main__':prepare_inputs()
