from pathlib import Path
import sys,json,hashlib,csv
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[2]/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
import joblib
d=pd.read_csv(P/'inputs.csv').set_index('ID');q=pd.read_csv(P/'predictions.csv').set_index('ID');protocol=json.loads((P/'protocol.json').read_text());cols=protocol['features']
assert len(d)==len(q)==129 and d.index.is_unique and q.index.is_unique
source=P.parent/'20261030-older44-sppb-timeseries/candidate-audit'
labels={(r['ID'],int(r['day'])):float(r['tft_tug']) for r in csv.DictReader((source/'opencap129-participant.csv').open()) if r['tft_tug']}
features={(r['ID'],int(r['day'])):[float(r[c]) for c in cols] for r in csv.DictReader((source/'opencap129-features.csv').open())}
for ident,r in d.iterrows():
    assert r.tft_tug==labels[ident,int(r.day)]
    assert np.allclose(r[cols].to_numpy(float),features[ident,int(r.day)])
seen=[]
for f in json.loads((P/'splits.json').read_text()):
    tr,te=f['train'],f['test'];assert not set(tr)&set(te);assert set(tr)|set(te)==set(d.index);seen+=te
    for inner in f['inner']:
        assert not set(inner['train'])&set(inner['test'])
        assert set(inner['train'])|set(inner['test'])==set(tr)
    for m in ['Ridge','RF']:
        loaded=joblib.load(P/f'{m}-{f["fold"]}.joblib')
        assert np.allclose(loaded.predict(d.loc[te,cols].to_numpy()),q.loc[te,m],atol=1e-10)
    assert np.allclose(q.loc[te,'Median'],np.median(d.loc[tr,'tft_tug']))
assert sorted(seen)==sorted(d.index)
s=json.loads((P/'summary.json').read_text())
for m in ['Mean','Median','Ridge','RF']:
    assert abs(float(np.mean(abs(q[m]-q.tft_tug)))-s['metrics'][m]['MAE_seconds'])<1e-10
assert all(c.startswith('10mwt_') for c in cols)
result={'passed':True,'people':129,'raw_target_and_feature_linkage':True,'person_outer_inner_disjoint_and_complete':True,'all10_models_replayed':True,'metrics_recomputed':True,'walking_only8_features':True,'code_sha256':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['run.py','verify.py','protocol.json']}}
(P/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
