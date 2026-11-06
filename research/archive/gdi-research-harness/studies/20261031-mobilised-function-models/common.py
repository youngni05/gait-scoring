from pathlib import Path
import os,sys,json,hashlib
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
SRC=ROOT/'gdi-research-harness/studies/20261031-mobilised-uncertainty-review'
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-mobilised-function-models')
SPEED=['ws_1030_avg_w','ws_10_p90_w','ws_30_avg_w','ws_30_p90_w','ws_30_var_w']
PATTERN=['strlen_1030_avg_w','strlen_30_avg_w','strlen_30_var_w','cadence_all_avg_w','cadence_all_var_w','cadence_30_avg_w','cadence_30_p90_w','strdur_all_avg_w','strdur_all_var_w','strdur_30_avg_w']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
