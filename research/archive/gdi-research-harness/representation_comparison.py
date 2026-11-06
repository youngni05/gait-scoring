"""Fixed Ridge comparison on the pinned development packet, no old test data."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import harness as h
from temporal_readiness import sha, ROOT, DATA
from weighted_ridge_job import metrics

CACHE = ROOT / "control/desktop-sessions/20261026T011215Z-a3a6e6"
DEVELOPMENT = ROOT / "jobs/development-baseline/runs/20261025T044558572569Z-71adb0"
METHODS = ("global_mean", "time_thirds", "leg_mean", "right_left")
DIMENSIONS = (512, 1536, 512, 1024)


def imports():
    sys.path.insert(0, str(DATA / "deps"))
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    return np, Ridge, StandardScaler, threadpool_limits


def load_inputs(folder):
    np, *_ = imports()
    h.require((folder / "plan.md").is_file(), "Missing prior plan")
    receipt = h.load(CACHE / "receipt.json")
    h.require(receipt["status"] == "completed" and sha(CACHE / "response.json") == receipt["response_sha256"], "Invalid cache receipt")
    for name in ("development_features.npz", "cache_manifest.json", "cache_result.json", "verification.json"):
        artifact = next(a for a in receipt["artifacts"] if a["path"] == (CACHE / name).relative_to(ROOT).as_posix())
        h.require(sha(CACHE / name) == artifact["sha256"], "Cache artifact hash mismatch")
    dev_receipt = h.load(DEVELOPMENT / "receipt.json")
    h.require(dev_receipt["status"] == "completed" and sha(DEVELOPMENT / "result.json") == dev_receipt["result_sha256"], "Invalid development receipt")
    previous = h.load(DEVELOPMENT / "result.json")
    packet = DEVELOPMENT / "development_packet.npz"
    h.require(sha(packet) == previous["artifacts"][0]["sha256"], "Packet hash mismatch")
    tracked = [CACHE / n for n in ("receipt.json", "response.json", "development_features.npz", "cache_manifest.json", "cache_result.json", "verification.json")]
    tracked += [packet, DEVELOPMENT / "result.json", DEVELOPMENT / "receipt.json", Path(__file__), ROOT / "weighted_ridge_job.py"]
    before = {str(p): sha(p) for p in tracked}
    manifest_file = folder / "comparison_sources.json"
    if manifest_file.exists():
        h.require(h.load(manifest_file)["sha256"] == before, "Frozen comparison inputs/code changed")
    else:
        h.save(manifest_file, {"sha256": before, "expected_baseline_mae": previous["validation_person_mae"]})
    with np.load(packet, allow_pickle=False) as p, np.load(CACHE / "development_features.npz", allow_pickle=False) as c:
        people, videos, split, y = (p[k].copy() for k in ("patient_ids", "video_ids", "split", "Y"))
        for k, actual in (("patient_ids", people), ("video_ids", videos), ("split", split)):
            np.testing.assert_array_equal(c[k], actual)
        np.testing.assert_array_equal(c["global_mean"], p["X"])
        features = {m: c[m].astype(np.float64) for m in METHODS}
    h.require(y.shape == (243, 2) and y.dtype == np.float64 and np.isfinite(y).all(), "Invalid labels")
    for m, d in zip(METHODS, DIMENSIONS):
        h.require(features[m].shape == (243, d) and np.isfinite(features[m]).all(), "Invalid features")
    train, val = split == "train", split == "validation"
    h.require(set(split) == {"train", "validation"} and len(set(videos)) == 243, "Invalid split or duplicate record")
    h.require((train.sum(), val.sum(), len(set(people[train])), len(set(people[val]))) == (203, 40, 175, 37), "Frozen split changed")
    h.require(not (set(people[train]) & set(people[val])), "Person overlap")
    return features, y, people, videos, split, before, previous


def require_unchanged(before):
    h.require(all(sha(Path(p)) == value for p, value in before.items()), "Source changed during calculation")


def fit_one(x, y, train, val):
    np, Ridge, StandardScaler, limits = imports()
    with limits(limits=4):
        scaler = StandardScaler().fit(x[train])
        xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
        model = Ridge(alpha=1.0, solver="svd").fit(xt, y[train])
        prediction = model.predict(xv)
        # Independent dual equation includes the model's intercept centering.
        xm, ym = xt.mean(axis=0), y[train].mean(axis=0)
        a, b = xt - xm, y[train] - ym
        dual = (xv - xm) @ a.T @ np.linalg.solve(a @ a.T + np.eye(len(a)), b) + ym
        h.require(np.allclose(prediction, dual, rtol=0, atol=1e-7), "Independent Ridge equation mismatch")
        singular = np.linalg.svd(a, compute_uv=False)
    h.require(np.allclose(scaler.mean_, x[train].mean(axis=0), rtol=0, atol=1e-12), "Scaler not train-only")
    details = {"feature_dimensions": x.shape[1], "training_record_mae": float(np.abs(model.predict(xt) - y[train]).mean()),
               "effective_degrees_of_freedom_excluding_intercept": float(np.sum(singular**2 / (singular**2 + 1.0))),
               "independent_prediction_max_abs_difference": float(np.max(np.abs(prediction - dual)))}
    parameters = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "coef": model.coef_, "intercept": model.intercept_}
    return prediction, parameters, details


def run(session):
    started = time.perf_counter()
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents, "Invalid session path")
    h.require(not (folder / "comparison_result.json").exists(), "Comparison already saved; do not retrain")
    features, y, people, videos, split, before, previous = load_inputs(folder)
    np, *_ = imports()
    train, val = split == "train", split == "validation"
    predictions, all_parameters, results, person_errors = {}, {}, {}, {}
    for method in METHODS:
        pred, params, diagnostic = fit_one(features[method], y, train, val)
        score, error = metrics(y[val], pred, people[val])
        if method == "global_mean":
            h.require(abs(score["person_mae"] - previous["validation_person_mae"]) < 1e-9, "Original validation MAE not reproduced")
        predictions[method], person_errors[method] = pred, error
        results[method] = {**score, **diagnostic}
        all_parameters.update({method + "__" + k: v for k, v in params.items()})
    model_file = folder / "representation_models.npz"
    np.savez_compressed(model_file, **all_parameters)
    with np.load(model_file, allow_pickle=False) as saved:
        for method in METHODS:
            restored = (features[method][val] - saved[method + "__scaler_mean"]) / saved[method + "__scaler_scale"]
            restored = restored @ saved[method + "__coef"].T + saved[method + "__intercept"]
            np.testing.assert_allclose(restored, predictions[method], rtol=0, atol=1e-10)
    pred_file = folder / "representation_validation_predictions.csv"
    with pred_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "video_id", "actual_L", "actual_R"] + [m + "_" + side for m in METHODS for side in ("L", "R")])
        for i, row in enumerate(np.flatnonzero(val)):
            writer.writerow([people[row], videos[row], *y[row], *[v for m in METHODS for v in predictions[m][i]]])
    comparisons = {}
    for name, candidate, reference in (("time_vs_global", "time_thirds", "global_mean"),
                                        ("legs_vs_global", "leg_mean", "global_mean"),
                                        ("sides_vs_legs", "right_left", "leg_mean")):
        delta = person_errors[reference] - person_errors[candidate]
        comparisons[name] = {"reference": reference, "candidate": candidate,
                             "person_mae_reduction": float(delta.mean()),
                             "people_improved": int((delta > 0).sum()), "people_worsened": int((delta < 0).sum()),
                             "people_equal": int((delta == 0).sum())}
    require_unchanged(before)
    result = {"status": "passed", "methods": results, "comparisons": comparisons, "alpha": 1.0,
              "solver": "svd", "dtype_for_all_score_models": "float64", "train_people": 175, "train_records": 203,
              "validation_people": 37, "validation_records": 40, "previous_test_records_used": 0,
              "model_weights_finetuned": False, "hyperparameter_search": False, "independent_final_evaluation": False,
              "source_files_unchanged": True, "saved_model_reproduction_passed": True,
              "elapsed_seconds": time.perf_counter() - started,
              "versions": {"numpy": np.__version__, "sklearn": __import__("sklearn").__version__}}
    h.save(folder / "comparison_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.session), ensure_ascii=True, indent=2))
    except Exception as error:
        import traceback
        folder = ROOT / "control/desktop-sessions" / args.session
        h.save(folder / "comparison_failure.json", {"error_type": type(error).__name__, "details": traceback.format_exc()})
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "details": "saved locally; no automatic retry"}))
        sys.exit(1)
