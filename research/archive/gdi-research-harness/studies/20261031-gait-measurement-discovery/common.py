"""Local, person-level measurement discovery with nested selection."""
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
ROOT = STUDIES.parents[1]
sys.path.insert(0, str(ROOT / 'outputs/gdi-ridge-experiment-20261016/deps'))
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

OUT = Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gait-measurement-discovery')
DATASETS = ['mobilise_sppb', 'mobilise_tug', 'mobilise_nonwalk', 'opencap_tug', 'opencap_patterns_tug', 'older44_sppb']
SPEED = ['ws_1030_avg_w', 'ws_10_p90_w', 'ws_30_avg_w', 'ws_30_p90_w', 'ws_30_var_w']
STRIDE = ['strlen_1030_avg_w', 'strlen_30_avg_w', 'strlen_30_var_w']
CADENCE = ['cadence_all_avg_w', 'cadence_all_var_w', 'cadence_30_avg_w', 'cadence_30_p90_w']
CYCLE = ['strdur_all_avg_w', 'strdur_all_var_w', 'strdur_30_avg_w']
SEEDS = [17, 41, 89]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load(name):
    d = dict(np.load(OUT/name/'input.npz', allow_pickle=False))
    meta = json.loads((OUT/name/'metadata.json').read_text(encoding='utf-8'))
    return d, meta


def estimator(model, n, seed):
    final = (Ridge(alpha=10) if model == 'Ridge' else RandomForestRegressor(
        n_estimators=128, min_samples_leaf=10 if n >= 500 else 2,
        max_features=1.0, random_state=seed, n_jobs=2))
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), StandardScaler(), final)


def fit_predict(model, x, y, train, test, seed, population_n):
    if model == 'Median':
        return np.full(len(test), np.median(y[train]))
    fit = estimator(model, population_n, seed)
    if model == 'RF':
        fit.fit(x[train], y[train], randomforestregressor__sample_weight=np.ones(len(train)))
    else:
        fit.fit(x[train], y[train])
    return fit.predict(x[test])


def family_sets(groups):
    families = list(groups)
    combos = []
    sizes = range(1, len(families)+1) if len(families) <= 5 else [1, 2, len(families)-1, len(families)]
    for count in sizes:
        for names in itertools.combinations(families, count):
            columns = sorted({i for name in names for i in groups[name]})
            combos.append(('+'.join(names), columns))
    return combos


def outer_folds(cohort, repeat):
    n = len(cohort)
    if len(set(cohort)) > 1 and min(pd.Series(cohort).value_counts()) >= 5:
        split = StratifiedKFold(5, shuffle=True, random_state=SEEDS[repeat])
        pairs = split.split(np.zeros(n), cohort)
    else:
        pairs = KFold(5, shuffle=True, random_state=SEEDS[repeat]).split(np.arange(n))
    result = np.empty(n, int)
    for fold, (_, test) in enumerate(pairs):
        result[test] = fold
    return result
