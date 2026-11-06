"""Paired straight-walking feature comparison; no chair or turning inputs."""
from pathlib import Path
import sys, json, hashlib
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
SOURCE = HERE.parent / '20261030-opencap415-tug-timeseries'
STABILITY = HERE.parent / '20261030-opencap415-speed-stability'
LABELS = HERE.parent / '20261030-older44-sppb-timeseries/candidate-audit/nmd_opencap_participant_info.csv'
SEEDS = [20261030, 20261031, 20261032]
PATTERN_NAMES = ['cycle_seconds', 'cycle_cv', 'cycle_time_asymmetry',
                 'knee_cycle_rom', 'knee_cycle_rom_asymmetry',
                 'hip_cycle_rom', 'hip_cycle_rom_asymmetry',
                 'ankle_cycle_rom', 'ankle_cycle_rom_asymmetry',
                 'hip_knee_cycle_correlation', 'pelvis_list_rom',
                 'lumbar_extension_rom', 'lumbar_bending_rom', 'lumbar_rotation_rom']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, obj):
    p = HERE / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')

def speed_features(x):
    v = x[17]
    return np.array([v.mean(), v.std(), np.abs(np.diff(v)).mean()])

def patterns(x):
    """Knee-flexion maxima define cycle proxies, not validated heel-strike times."""
    x = np.asarray(x, dtype=float); n = x.shape[1]
    # Sharp angle jumps affect neighboring derivative/smoothing samples as well.
    bad = np.any(np.abs(np.diff(x[:17], axis=1)) * 30 > 1000, axis=0)
    bad = np.r_[bad, False] | np.r_[False, bad]
    bad = np.convolve(bad.astype(int), np.ones(7), mode='same') > 0
    smooth = np.stack([np.convolve(np.pad(a, (2,2), mode='edge'), np.ones(5)/5, mode='valid') for a in x[:17]])
    intervals = []; spans = []; peak_lists = []
    for knee in [5, 11]:
        peaks, _ = find_peaks(smooth[knee], distance=12, prominence=8)
        peaks = peaks[(peaks >= 2) & (peaks < n-2)]
        valid = [(int(a),int(b)) for a,b in zip(peaks[:-1],peaks[1:])
                 if .6 <= (b-a)/30 <= 2.5 and not bad[a:b+1].any()]
        spans.append(valid); intervals.append(np.array([(b-a)/30 for a,b in valid]))
        peak_lists.append(peaks.tolist())
    out = np.full(len(PATTERN_NAMES), np.nan)
    both = all(len(a) >= 1 for a in intervals)
    if both:
        durations = np.concatenate(intervals)
        out[0] = np.median(durations)
        if all(len(a) >= 2 for a in intervals): out[1] = durations.std()/durations.mean()
        med = [np.median(a) for a in intervals]
        out[2] = abs(med[0]-med[1])/np.mean(med)
        for j, channels in enumerate([[5,11],[2,8],[6,12]]):
            rom = [np.median([np.ptp(smooth[c,a:b+1]) for a,b in spans[s]]) for s,c in enumerate(channels)]
            out[3+2*j] = np.mean(rom)
            if np.mean(rom) > 1: out[4+2*j] = abs(rom[0]-rom[1])/np.mean(rom)
        corrs = []
        for side,(hip,knee) in enumerate([(2,5),(8,11)]):
            for a,b in spans[side]:
                h,k = smooth[hip,a:b+1],smooth[knee,a:b+1]
                if h.std() > 1 and k.std() > 1: corrs.append(np.corrcoef(h,k)[0,1])
        if corrs: out[9] = np.mean(corrs)
    # Only stable samples inside the exact input window; no padding with other frames.
    good = ~bad
    if good.mean() >= .8:
        for j,c in enumerate([1,14,15,16]):
            out[10+j] = np.diff(np.percentile(smooth[c,good], [5,95]))[0]
    return out, {'bad_fraction': float(bad.mean()), 'right_cycles': len(spans[0]),
                 'left_cycles': len(spans[1]), 'cycle_available': bool(both),
                 'cv_available': bool(np.isfinite(out[1])), 'peaks':peak_lists,'spans':spans}

def make_model(family, setting, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    model = (RandomForestRegressor(n_estimators=128, min_samples_leaf=int(setting),
                                  max_features=1., random_state=seed, n_jobs=1)
             if family == 'RF' else Ridge(alpha=float(setting)))
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                         StandardScaler(), model)

def load_data(name):
    return dict(np.load(HERE / name / 'input.npz', allow_pickle=False))
