from pathlib import Path
import sys,json,hashlib,importlib.util
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PRIOR=HERE.parent/'20261031-opencap-walk-pattern'
QUALITY=HERE.parent/'20261031-opencap-walk-quality'
SOURCE=HERE.parent/'20261030-opencap415-tug-timeseries'
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
spec=importlib.util.spec_from_file_location('quality_features',QUALITY/'common.py')
quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)
LABELS=quality.prior.LABELS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def load():return dict(np.load(PRIOR/'STS_common6/input.npz',allow_pickle=False))
