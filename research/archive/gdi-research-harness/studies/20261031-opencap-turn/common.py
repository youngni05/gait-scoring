from pathlib import Path
import sys,json,hashlib,io,re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=HERE.parent/'20261030-opencap415-tug-timeseries'
PREV=HERE.parent/'20261030-opencap415-speed-stability'
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
SEEDS=[20261030,20261031,20261032]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 p=HERE/n;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def read(p):
 text=p.read_text();assert 'inDegrees=yes' in text
 frame=pd.read_csv(io.StringIO(text.split('endheader',1)[1].strip()),sep=r'\s+')
 if not np.isfinite(frame.to_numpy(dtype=float)).all():raise ValueError('nonfinite_or_incomplete_source_row')
 return frame.to_records(index=False)
def identity(name):
 p,v,s=re.search(r'(sub-\d+)/visit-(\d+)/ses-(\d+)',name).groups()
 return p,int(v),int(s)
