"""Frozen MotionBERT-Lite probe, with a fixed person-disjoint held-out test.

Run locally using the experiment's isolated dependencies. This is a predictive
pilot on the retained Stanford records, not a clinical validation study.
"""
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/gdi-ridge-experiment-20261016"
MB = ROOT / "outputs/motionbert-smoke-20261015"
for path in [MB / "vendor", MB / "deps", OUT / "deps"]:
    sys.path.insert(0, str(path))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import torch
from lib.model.DSTformer import DSTformer
from lib.utils.utils_data import crop_scale
import joblib
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 20261016
ALPHAS = [0.01, 0.1, 1., 10., 100., 1000., 10000.]
WEIGHT_HASH = "6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461"


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def peak_memory_bytes():
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ["PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
    handle_fn = ctypes.windll.kernel32.GetCurrentProcess
    handle_fn.restype = ctypes.c_void_p
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    assert fn(handle_fn(), ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize)


def person_errors(y, pred, people):
    errors = np.abs(y - pred)
    ids = sorted(set(people))
    return np.stack([errors[people == p].mean(axis=0) for p in ids])


def metrics(y, pred, people):
    errors = person_errors(y, pred, people)
    return {"person_macro_MAE": float(errors.mean()),
            "person_MAE_left": float(errors[:, 0].mean()),
            "person_MAE_right": float(errors[:, 1].mean()),
            "record_MAE": float(np.abs(y-pred).mean()),
            "people": len(errors), "records": len(y)}


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    prepared = json.loads((OUT / "prepared.json").read_text(encoding="utf-8"))
    records = [r for r in prepared["records"] if r["status"] == "included"]
    people = np.array([str(r["Patient_ID"]) for r in records])
    unique = np.array(sorted(set(people)))
    shuffled = np.random.default_rng(SEED).permutation(unique)
    ntrain, nval = int(0.70*len(unique)), int(0.15*len(unique))
    groups = {"train": shuffled[:ntrain], "validation": shuffled[ntrain:ntrain+nval],
              "test": shuffled[ntrain+nval:]}
    assert all(not set(groups[a]) & set(groups[b]) for a,b in
               [("train","validation"),("train","test"),("validation","test")])
    masks = {name: np.isin(people, ids) for name,ids in groups.items()}
    assert np.all(sum(m.astype(int) for m in masks.values()) == 1)
    config = {"seed": SEED, "sides": ["L", "R"], "alpha_grid": ALPHAS,
        "selection_metric": "validation mean absolute error, equal weight per person and side",
        "fit": "unweighted records; scaler fit train only, Ridge fit train only",
        "test_used_for_selection": False, "feature_pooling": "mean over frames and joints",
        "batch_size": 1, "cpu_threads": 4, "window_frames": 243,
        "split": {name: {"people":len(ids), "records":int(masks[name].sum()),
                         "patient_ids":ids.tolist()} for name,ids in groups.items()},
        "frozen_encoder": True, "weights_sha256": WEIGHT_HASH,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # Fix and persist the protocol before extracting features or fitting models.
    write_json("protocol.json", config)

    weights = MB / "assets/MB_lite.bin"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == WEIGHT_HASH
    model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                     num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                     qkv_bias=True, att_fuse=True)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k.removeprefix("module."):v for k,v in checkpoint["model_pos"].items()}, strict=True)
    model.requires_grad_(False)
    model.eval()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    (OUT / "features").mkdir(exist_ok=True)
    vectors, inference_times = [], []
    feature_started = time.perf_counter()
    for i, record in enumerate(records):
        path = OUT / "features" / f"{record['video_id']}.npy"
        x = np.load(OUT / record["input_file"])["input"]
        assert x.dtype == np.float32 and x.shape == (243,17,3) and np.isfinite(x).all()
        if not path.exists():
            tick = time.perf_counter()
            with torch.inference_mode():
                rep = model.get_representation(torch.from_numpy(x[None]))
                assert rep.shape == (1,243,17,512) and torch.isfinite(rep).all()
                vector = rep.mean(dim=(1,2)).numpy()[0]
            inference_times.append(time.perf_counter()-tick)
            np.save(path, vector)
        vector = np.load(path)
        assert vector.shape == (512,) and np.isfinite(vector).all()
        vectors.append(vector)
        if record["video_id"] == "10576905":
            # Independent existing sample verifies both preprocessing and features.
            previous = ROOT / "outputs/stanford-sample-check-20261016"
            old_x = np.load(previous / "input_h36m.npy")[0]
            np.testing.assert_allclose(x, old_x, rtol=1e-6, atol=2e-7)
            np.testing.assert_allclose(vector, np.load(previous / "feature_vector.npy"), rtol=1e-5, atol=2e-6)
        if (i+1)%25 == 0 or i+1 == len(records):
            print(json.dumps({"feature_records":i+1,"total":len(records),
                  "elapsed_seconds":round(time.perf_counter()-feature_started,1)}),flush=True)
    feature_seconds = time.perf_counter()-feature_started
    X = np.stack(vectors).astype(np.float64)
    Y = np.array([[r["gdi"]["L"],r["gdi"]["R"]] for r in records],np.float64)
    np.savez_compressed(OUT / "feature_table.npz", X=X, Y=Y, patient_ids=people,
                        video_ids=np.array([r["video_id"] for r in records]))

    train, val, test = [masks[n] for n in ["train","validation","test"]]
    fit_started = time.perf_counter()
    scaler = StandardScaler().fit(X[train])
    np.testing.assert_allclose(scaler.mean_, X[train].mean(axis=0))
    xt = scaler.transform(X[train])
    xv = scaler.transform(X[val])
    trials = []
    fitted = []
    with threadpool_limits(limits=4):
        for alpha in ALPHAS:
            head = Ridge(alpha=alpha, solver="svd").fit(xt,Y[train])
            score = metrics(Y[val],head.predict(xv),people[val])
            trials.append({"alpha":alpha, **score})
            fitted.append(head)
    choice = min(range(len(trials)),key=lambda i:trials[i]["person_macro_MAE"])
    head = fitted[choice]
    mean_gdi = Y[train].mean(axis=0)
    # The untouched test is first evaluated only after selecting alpha on validation.
    test_prediction = head.predict(scaler.transform(X[test]))
    predictions = head.predict(scaler.transform(X))
    np.testing.assert_allclose(predictions[test],test_prediction)
    baseline = np.broadcast_to(mean_gdi,Y.shape)
    results = {name:{"ridge":metrics(Y[mask],predictions[mask],people[mask]),
                     "training_mean_baseline":metrics(Y[mask],baseline[mask],people[mask])}
                     for name,mask in masks.items()}
    ridge_errors = person_errors(Y[test],predictions[test],people[test]).mean(axis=1)
    baseline_errors = person_errors(Y[test],baseline[test],people[test]).mean(axis=1)
    improvements = baseline_errors-ridge_errors
    rng = np.random.default_rng(SEED+1)
    boot = improvements[rng.integers(len(improvements),size=(10000,len(improvements)))].mean(axis=1)
    ci = np.percentile(boot,[2.5,97.5]).tolist()
    joblib.dump({"scaler":scaler,"ridge":head,"sides":["L","R"],
                 "training_mean_gdi":mean_gdi,"protocol":config},OUT / "ridge_model.joblib")
    restored = joblib.load(OUT / "ridge_model.joblib")
    np.testing.assert_array_equal(restored["ridge"].predict(restored["scaler"].transform(X[test])),test_prediction)
    with (OUT / "predictions.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id","video_id","split","actual_L","actual_R","ridge_L","ridge_R","baseline_L","baseline_R"])
        for i,r in enumerate(records):
            split = next(n for n,m in masks.items() if m[i])
            writer.writerow([people[i],r["video_id"],split,*Y[i],*predictions[i],*mean_gdi])
    fit_seconds = time.perf_counter()-fit_started
    report = {"status":"frozen_encoder_ridge_held_person_test_completed",
        "prepared_records":prepared["records_included"],"prepared_people":prepared["people_included"],
        "quality_exclusions":prepared["exclusion_counts"], "selected_alpha":ALPHAS[choice],
        "validation_trials":trials,"metrics":results,
        "test_person_MAE_improvement":float(improvements.mean()),
        "test_person_MAE_improvement_bootstrap_95_CI":ci,
        "test_people_improved":int((improvements>0).sum()),
        "training_mean_GDI_L_R":mean_gdi.tolist(),
        "feature_extraction_seconds_this_run":feature_seconds,
        "new_inferences":len(inference_times),
        "inference_median_seconds":float(np.median(inference_times)) if inference_times else None,
        "head_training_selection_export_seconds":fit_seconds,
        "total_seconds":time.perf_counter()-started,
        "process_peak_working_set_bytes":peak_memory_bytes(),
        "runtime":{"torch":torch.__version__,"sklearn":sklearn.__version__,"numpy":np.__version__,
                   "device":"cpu","cuda_available":torch.cuda.is_available()},
        "checks":{"strict_weight_load":True,"encoder_frozen":True,
                  "person_disjoint":True,"test_used_for_model_selection":False,
                  "saved_model_reload_predictions_identical":True},
        "limitations":["One fixed split; no nested cross validation or external cohort validation",
            "Strict 243-frame window selection excludes many records and may bias the retained cohort",
            "No RGB visual verification of person identity or anatomical pose accuracy",
            "Some upper-body joints have zero confidence; no interpolation was performed",
            "Visit-level GDI, not a per-frame label; original paper cohort filters not reproduced",
            "A successful probe does not identify which physical gait property each latent coordinate means",
            "General older-adult, home-camera and clinical usefulness are not established"]}
    write_json("result.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
