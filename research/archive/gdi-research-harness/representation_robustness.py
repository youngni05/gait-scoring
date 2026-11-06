"""Fixed training-person folds and paired descriptive resampling, no tuning."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import time

import harness as h
from temporal_readiness import ROOT, sha
from representation_comparison import METHODS, imports, load_inputs, fit_one, require_unchanged
from weighted_ridge_job import metrics

PAIRS = (("time_vs_global", "time_thirds", "global_mean"),
         ("legs_vs_global", "leg_mean", "global_mean"),
         ("sides_vs_legs", "right_left", "leg_mean"))


def csv_errors(path):
    """Stdlib-only person reduction, independent from NumPy metrics."""
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    errors = {method: defaultdict(list) for method in METHODS}
    for row in rows:
        for method in METHODS:
            error = statistics.mean(abs(float(row[method + "_" + side]) - float(row["actual_" + side])) for side in ("L", "R"))
            errors[method][row["patient_id"]].append(error)
    return {m: {p: statistics.mean(e) for p, e in people.items()} for m, people in errors.items()}


def run(session):
    started = time.perf_counter()
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents, "Invalid session")
    h.require(not (folder / "robustness_result.json").exists(), "Robustness results already saved; no repeated fits")
    h.require("2단계 계획" in (folder / "plan.md").read_text(encoding="utf-8"), "Missing prior stage plan")
    features, y, people, videos, split, before, _ = load_inputs(folder)
    np, *_ = imports()
    comparison = h.load(folder / "comparison_result.json")
    own_sources = [folder / "comparison_result.json", folder / "representation_models.npz",
                   folder / "representation_validation_predictions.csv", Path(__file__)]
    stage_before = {str(p): sha(p) for p in own_sources}
    errors = csv_errors(folder / "representation_validation_predictions.csv")
    for method in METHODS:
        h.require(len(errors[method]) == 37, "Validation person count mismatch")
        h.require(abs(statistics.mean(errors[method].values()) - comparison["methods"][method]["person_mae"]) < 1e-10, "Independent CSV MAE mismatch")
    person_order = sorted(errors[METHODS[0]])
    for method in METHODS:
        h.require(sorted(errors[method]) == person_order, "Paired validation people mismatch")
    indices = np.random.default_rng(20261026).integers(0, 37, size=(10000, 37))
    bootstrap = {}
    for name, candidate, reference in PAIRS:
        delta = np.array([errors[reference][p] - errors[candidate][p] for p in person_order])
        samples = delta[indices].mean(axis=1)
        low, high = np.percentile(samples, [2.5, 97.5])
        bootstrap[name] = {"mean_person_mae_reduction": float(delta.mean()), "interval_percentile_95": [float(low), float(high)],
                           "median_person_reduction": float(np.median(delta)), "contains_zero": bool(low <= 0 <= high)}
    h.save(folder / "paired_resampling.json", {"seed": 20261026, "resamples": 10000, "people": 37,
             "unit": "person, all records already averaged within person", "multiple_comparison_adjusted": False,
             "independent_final_evaluation": False, "comparisons": bootstrap})

    # Work only inside the original training partition; original validation remains untouched.
    train = split == "train"
    xt = {m: x[train].copy() for m, x in features.items()}
    yt, pt, vt = y[train].copy(), people[train].copy(), videos[train].copy()
    shuffled = np.array(sorted(set(pt)))
    np.random.default_rng(20261026).shuffle(shuffled)
    groups = np.array_split(shuffled, 5)
    h.require(all(len(g) == 35 for g in groups), "Wrong fold sizes")
    h.save(folder / "internal_cv_protocol.json", {"seed": 20261026, "folds": 5, "source_partition": "original train only",
            "people_per_fold": 35, "alpha": 1.0, "solver": "svd", "methods": list(METHODS),
            "fold_people": [[str(p) for p in group] for group in groups], "source_sha256": {**before, **stage_before}})
    out = {m: np.full((203, 2), np.nan) for m in METHODS}
    visits, fold_ids = np.zeros(203, dtype=int), np.full(203, -1, dtype=int)
    folds = []
    for fold, group in enumerate(groups):
        val = np.isin(pt, group)
        fit = ~val
        h.require(len(set(pt[fit])) == 140 and len(set(pt[val])) == 35 and not (set(pt[fit]) & set(pt[val])), "Internal person overlap")
        fold_result = {"fold": fold, "training_people": 140, "held_people": 35,
                       "training_records": int(fit.sum()), "held_records": int(val.sum()), "methods": {}}
        for method in METHODS:
            prediction, _, diagnostic = fit_one(xt[method], yt, fit, val)
            out[method][val] = prediction
            score, _ = metrics(yt[val], prediction, pt[val])
            fold_result["methods"][method] = {**score, **diagnostic}
        folds.append(fold_result)
        visits[val] += 1
        fold_ids[val] = fold
        h.save(folder / "internal_cv_progress.json", {"completed_folds": folds})
    h.require((visits == 1).all() and (fold_ids >= 0).all(), "OOF coverage mismatch")
    aggregate = {m: metrics(yt, out[m], pt)[0] for m in METHODS}
    comparisons = {}
    for name, candidate, reference in PAIRS:
        changes = [f["methods"][reference]["person_mae"] - f["methods"][candidate]["person_mae"] for f in folds]
        comparisons[name] = {"oof_person_mae_reduction": aggregate[reference]["person_mae"] - aggregate[candidate]["person_mae"],
                             "fold_reductions": changes, "folds_improved": sum(c > 0 for c in changes),
                             "folds_worsened": sum(c < 0 for c in changes)}
    csv_file = folder / "training_internal_oof_predictions.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "video_id", "fold", "actual_L", "actual_R"] + [m + "_" + s for m in METHODS for s in ("L", "R")])
        for i in range(203):
            writer.writerow([pt[i], vt[i], fold_ids[i], *yt[i], *[v for m in METHODS for v in out[m][i]]])
    independent = csv_errors(csv_file)
    for method in METHODS:
        h.require(len(independent[method]) == 175, "Wrong OOF person count")
        h.require(abs(statistics.mean(independent[method].values()) - aggregate[method]["person_mae"]) < 1e-10, "Independent OOF MAE mismatch")
    require_unchanged(before)
    require_unchanged(stage_before)
    result = {"status": "passed", "internal_cv": {"seed": 20261026, "folds": 5, "people": 175, "records": 203,
               "original_validation_records_used_for_cv": 0, "methods": aggregate, "comparisons": comparisons},
              "validation_paired_bootstrap": bootstrap, "independent_csv_metrics_passed": True,
              "each_training_record_predicted_once": True, "source_files_unchanged": True,
              "old_test_records_used": 0, "hyperparameter_search": False,
              "elapsed_seconds": time.perf_counter() - started}
    h.save(folder / "robustness_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.session), ensure_ascii=True, indent=2))
    except Exception as error:
        import traceback
        h.save(ROOT / "control/desktop-sessions" / args.session / "robustness_failure.json",
               {"error_type": type(error).__name__, "details": traceback.format_exc()})
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "details": "saved locally; no automatic retry"}))
        raise SystemExit(1)
