from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
