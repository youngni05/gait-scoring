"""Match Ridge effective degrees of freedom from training X only, never labels."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import harness as h
from temporal_readiness import ROOT, sha
from representation_comparison import METHODS, imports, load_inputs, require_unchanged
from representation_robustness import csv_errors, PAIRS
from weighted_ridge_job import metrics


def alpha_for_df(squared_singular, target):
    """Only singular values and a training-derived target enter this selection."""
    np, *_ = imports()
    h.require(target > 0 and target < (squared_singular > 0).sum(), "Unreachable target df")
    def df(alpha):
        return float(np.sum(squared_singular / (squared_singular + alpha)))
    low, high = 0.0, 1.0
    while df(high) > target:
        high *= 2
        h.require(high < 1e20, "No finite alpha bracket")
    for _ in range(100):
        middle = (low + high) / 2
        if df(middle) > target:
            low = middle
        else:
            high = middle
    alpha = (low + high) / 2
    h.require(abs(df(alpha) - target) < 1e-7, "Target df not matched")
    return alpha, df(alpha)


def design(x, fit):
    np, _, StandardScaler, limits = imports()
    scaler = StandardScaler().fit(x[fit])
    xt = scaler.transform(x[fit])
    centered = xt - xt.mean(axis=0)
    with limits(limits=4):
        squared = np.linalg.svd(centered, compute_uv=False)**2
    return scaler, xt, centered, squared


def fit_matched(x, y, fit, held, target):
    np, Ridge, _, limits = imports()
    scaler, xt, a, squared = design(x, fit)
    alpha, actual_df = alpha_for_df(squared, target)
    # Parameter selection is complete before score labels enter the estimator.
    with limits(limits=4):
        model = Ridge(alpha=alpha, solver="svd").fit(xt, y[fit])
        xv = scaler.transform(x[held])
        pred = model.predict(xv)
        gram = a @ a.T
        inverse_system = gram + alpha * np.eye(len(a))
        trace_df = float(np.trace(np.linalg.solve(inverse_system, gram)))
        ym = y[fit].mean(axis=0)
        independent = (xv - xt.mean(axis=0)) @ a.T @ np.linalg.solve(inverse_system, y[fit] - ym) + ym
    h.require(abs(trace_df - target) < 1e-7, "Independent trace df mismatch")
    h.require(np.allclose(pred, independent, rtol=0, atol=1e-7), "Independent matched Ridge mismatch")
    np.testing.assert_allclose(scaler.mean_, x[fit].mean(axis=0), rtol=0, atol=1e-12)
    parameters = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "coef": model.coef_, "intercept": model.intercept_}
    return pred, parameters, {"alpha_from_training_X_only": alpha, "target_df": target, "matched_df": actual_df,
            "independent_trace_df": trace_df, "independent_prediction_max_abs_difference": float(np.max(np.abs(pred - independent)))}


def reference_csv(path, expected_videos):
    np, *_ = imports()
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    h.require([r["video_id"] for r in rows] == [str(v) for v in expected_videos], "Reference CSV row order mismatch")
    return np.array([[float(r["global_mean_L"]), float(r["global_mean_R"])] for r in rows])


def export(path, videos, people, y, predictions):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "video_id", "actual_L", "actual_R"] + [m + "_" + s for m in METHODS for s in ("L", "R")])
        for i in range(len(videos)):
            writer.writerow([people[i], videos[i], *y[i], *[v for m in METHODS for v in predictions[m][i]]])


def run(session):
    started = time.perf_counter()
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents, "Invalid session")
    h.require(not (folder / "complexity_result.json").exists(), "Complexity result already saved; no repeated fits")
    h.require("3단계 계획" in (folder / "plan.md").read_text(encoding="utf-8"), "Missing stage plan")
    features, y, people, videos, split, before, _ = load_inputs(folder)
    np, *_ = imports()
    stage_sources = [Path(__file__), ROOT / "representation_robustness.py"] + [folder / n for n in
            ("comparison_result.json", "robustness_result.json", "internal_cv_protocol.json", "representation_validation_predictions.csv", "training_internal_oof_predictions.csv")]
    stage_before = {str(p): sha(p) for p in stage_sources}
    h.save(folder / "complexity_sources.json", {"sha256": {**before, **stage_before},
             "selection_rule": "match global_mean alpha=1 effective df; each fit partition's X only; no labels or held scores"})
    # Analytical sanity check: three equal singular squares of 3 have df 1.8 at alpha 2.
    alpha, _ = alpha_for_df(np.array([3.0, 3.0, 3.0]), 1.8)
    h.require(abs(alpha - 2.0) < 1e-10, "Alpha solver analytical check failed")
    train, val = split == "train", split == "validation"
    _, _, _, squared = design(features["global_mean"], train)
    target = float(np.sum(squared / (squared + 1.0)))
    primary = h.load(folder / "comparison_result.json")
    h.require(abs(target - primary["methods"]["global_mean"]["effective_degrees_of_freedom_excluding_intercept"]) < 1e-10, "Reference target changed")
    vp = {"global_mean": reference_csv(folder / "representation_validation_predictions.csv", videos[val])}
    val_results = {"global_mean": primary["methods"]["global_mean"]}
    params = {}
    for method in METHODS[1:]:
        pred, parameter, details = fit_matched(features[method], y, train, val, target)
        vp[method] = pred
        val_results[method] = {**metrics(y[val], pred, people[val])[0], **details}
        params.update({method + "__" + k: v for k, v in parameter.items()})
    np.savez_compressed(folder / "matched_models.npz", **params)
    with np.load(folder / "matched_models.npz", allow_pickle=False) as saved:
        for method in METHODS[1:]:
            restored = (features[method][val] - saved[method + "__scaler_mean"]) / saved[method + "__scaler_scale"]
            np.testing.assert_allclose(restored @ saved[method + "__coef"].T + saved[method + "__intercept"], vp[method], rtol=0, atol=1e-10)
    export(folder / "matched_validation_predictions.csv", videos[val], people[val], y[val], vp)

    xt = {m: x[train] for m, x in features.items()}
    yt, pt, vt = y[train], people[train], videos[train]
    protocol = h.load(folder / "internal_cv_protocol.json")
    groups = protocol["fold_people"]
    h.require(len(groups) == 5 and len(set(p for g in groups for p in g)) == 175 and set(p for g in groups for p in g) == set(pt), "CV protocol mismatch")
    oof = {m: np.full((203, 2), np.nan) for m in METHODS}
    oof["global_mean"] = reference_csv(folder / "training_internal_oof_predictions.csv", vt)
    visits = np.zeros(203, dtype=int)
    folds = []
    for number, group in enumerate(groups):
        held = np.isin(pt, group)
        fit = ~held
        h.require(len(set(pt[held])) == 35 and len(set(pt[fit])) == 140 and not set(pt[fit]) & set(pt[held]), "CV person overlap")
        _, _, _, spectrum = design(xt["global_mean"], fit)
        target_fold = float(np.sum(spectrum / (spectrum + 1.0)))
        row = {"fold": number, "target_df": target_fold, "methods": {"global_mean": metrics(yt[held], oof["global_mean"][held], pt[held])[0]}}
        for method in METHODS[1:]:
            pred, _, details = fit_matched(xt[method], yt, fit, held, target_fold)
            oof[method][held] = pred
            row["methods"][method] = {**metrics(yt[held], pred, pt[held])[0], **details}
        visits[held] += 1
        folds.append(row)
    h.require((visits == 1).all() and all(np.isfinite(v).all() for v in oof.values()), "Matched OOF coverage mismatch")
    oof_results = {m: metrics(yt, oof[m], pt)[0] for m in METHODS}
    export(folder / "matched_training_oof_predictions.csv", vt, pt, yt, oof)
    for path, expected in ((folder / "matched_validation_predictions.csv", val_results), (folder / "matched_training_oof_predictions.csv", oof_results)):
        recalculated = csv_errors(path)
        for method in METHODS:
            h.require(abs(statistics.mean(recalculated[method].values()) - expected[method]["person_mae"]) < 1e-10, "Matched CSV metric mismatch")
    summaries = {}
    for name, candidate, reference in PAIRS:
        summaries[name] = {"validation_person_mae_reduction": val_results[reference]["person_mae"] - val_results[candidate]["person_mae"],
                          "internal_oof_person_mae_reduction": oof_results[reference]["person_mae"] - oof_results[candidate]["person_mae"],
                          "internal_folds_improved": sum(f["methods"][reference]["person_mae"] > f["methods"][candidate]["person_mae"] for f in folds)}
    require_unchanged(before)
    require_unchanged(stage_before)
    result = {"status": "passed", "validation": val_results, "internal_training_oof": oof_results,
              "comparisons": summaries, "fold_details": folds, "alpha_selected_from_training_inputs_only": True,
              "validation_performance_hyperparameter_search": False, "primary_alpha1_results_preserved": True,
              "same_fixed_splits": True, "old_test_records_used": 0, "source_files_unchanged": True,
              "analytical_alpha_solver_check": True, "independent_trace_and_ridge_checks": True,
              "independent_csv_metrics_passed": True, "saved_model_reproduction_passed": True,
              "elapsed_seconds": time.perf_counter() - started}
    h.save(folder / "complexity_result.json", result)
    return {k: v for k, v in result.items() if k != "fold_details"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.session), ensure_ascii=True, indent=2))
    except Exception as error:
        import traceback
        h.save(ROOT / "control/desktop-sessions" / args.session / "complexity_failure.json", {"error_type": type(error).__name__, "details": traceback.format_exc()})
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "details": "saved locally; no automatic retry"}))
        raise SystemExit(1)
