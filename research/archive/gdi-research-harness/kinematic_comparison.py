"""First fixed 16-angle-summary and 512+16 comparison, development data only."""
import argparse
import json
from pathlib import Path
import statistics
import time

import harness as h
from temporal_readiness import ROOT, DATA, sha
from representation_comparison import imports, load_inputs, fit_one, require_unchanged
from representation_complexity import reference_csv
from kinematic_readiness import summaries, FEATURE_NAMES
from weighted_ridge_job import metrics

NAMES = ("global_mean", "kinematic16", "combined528")


def run(session):
    import csv
    started = time.perf_counter()
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents, "Invalid session")
    h.require("5단계 계획" in (folder / "plan.md").read_text(encoding="utf-8"), "Missing stage plan")
    h.require(not (folder / "kinematic_comparison_result.json").exists(), "Comparison already saved; no repeated fits")
    features, y, people, videos, split, before, _ = load_inputs(folder)
    np, *_ = imports()
    windows = [DATA / "windows" / (str(video) + ".npz") for video in videos]
    stage_tracked = windows + [Path(__file__), ROOT / "kinematic_readiness.py", ROOT / "representation_complexity.py",
                     folder / "internal_cv_protocol.json", folder / "representation_validation_predictions.csv",
                     folder / "training_internal_oof_predictions.csv", folder / "three_training_kinematic_probes.npz"]
    stage_before = {str(p): sha(p) for p in stage_tracked}
    h.save(folder / "kinematic_comparison_sources.json", {"sha256": {**before, **stage_before}})
    values = []
    for index, path in enumerate(windows):
        with np.load(path, allow_pickle=False) as item:
            motion = item["input"].copy()
        h.require(motion.shape == (243, 17, 3) and (motion[:, 1:7, 2] > 0).all(), "Observation contract mismatch")
        try:
            feature, angle = summaries(motion)
        except ValueError:
            h.save(folder / "kinematic_invalid_row.json", {"row_index": index, "video_id": str(videos[index]), "reason": "angle feature contract violation; no row excluded"})
            raise
        h.require(feature.shape == (16,) and np.isfinite(feature).all() and ((angle >= 0) & (angle <= 180)).all(), "Invalid angle features")
        values.append(feature)
    kinematic = np.stack(values)
    chosen = sorted(np.flatnonzero(split == "train"), key=lambda i: str(videos[i]))[:3]
    with np.load(folder / "three_training_kinematic_probes.npz", allow_pickle=False) as probe:
        np.testing.assert_array_equal(kinematic[chosen], probe["features"])
    cache = folder / "kinematic_features.npz"
    np.savez_compressed(cache, features=kinematic, feature_names=np.array(FEATURE_NAMES), patient_ids=people, video_ids=videos, split=split)
    with np.load(cache, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["features"], kinematic)
        h.require("Y" not in saved.files, "Labels unexpectedly cached")
    source_features = {"kinematic16": kinematic, "combined528": np.concatenate([features["global_mean"], kinematic], axis=1)}
    train, val = split == "train", split == "validation"
    primary = h.load(folder / "comparison_result.json")
    vp = {"global_mean": reference_csv(folder / "representation_validation_predictions.csv", videos[val])}
    val_results = {"global_mean": primary["methods"]["global_mean"]}
    parameters = {}
    for name, x in source_features.items():
        pred, params, details = fit_one(x, y, train, val)
        vp[name] = pred
        val_results[name] = {**metrics(y[val], pred, people[val])[0], **details}
        parameters.update({name + "__" + k: v for k, v in params.items()})
    np.savez_compressed(folder / "kinematic_models.npz", **parameters)
    with np.load(folder / "kinematic_models.npz", allow_pickle=False) as saved:
        for name, x in source_features.items():
            restored = (x[val] - saved[name + "__scaler_mean"]) / saved[name + "__scaler_scale"]
            np.testing.assert_allclose(restored @ saved[name + "__coef"].T + saved[name + "__intercept"], vp[name], rtol=0, atol=1e-10)
    xt = {m: x[train] for m, x in source_features.items()}
    yt, pt, vt = y[train], people[train], videos[train]
    groups = h.load(folder / "internal_cv_protocol.json")["fold_people"]
    h.require(len(groups) == 5 and set(p for group in groups for p in group) == set(pt), "Wrong CV protocol")
    out = {name: np.full((203, 2), np.nan) for name in NAMES}
    out["global_mean"] = reference_csv(folder / "training_internal_oof_predictions.csv", vt)
    visits = np.zeros(203, dtype=int)
    folds = []
    for fold, group in enumerate(groups):
        held, fit = np.isin(pt, group), ~np.isin(pt, group)
        h.require(len(set(pt[held])) == 35 and len(set(pt[fit])) == 140 and not set(pt[held]) & set(pt[fit]), "Fold overlap")
        row = {"fold": fold, "methods": {"global_mean": metrics(yt[held], out["global_mean"][held], pt[held])[0]}}
        for name, x in xt.items():
            pred, _, details = fit_one(x, yt, fit, held)
            out[name][held] = pred
            row["methods"][name] = {**metrics(yt[held], pred, pt[held])[0], **details}
        visits[held] += 1
        folds.append(row)
    h.require((visits == 1).all() and all(np.isfinite(pred).all() for pred in out.values()), "OOF coverage mismatch")
    oof_results = {name: metrics(yt, pred, pt)[0] for name, pred in out.items()}
    for path, pp, vv, yy, predictions, expected in (
            (folder / "kinematic_validation_predictions.csv", people[val], videos[val], y[val], vp, val_results),
            (folder / "kinematic_training_oof_predictions.csv", pt, vt, yt, out, oof_results)):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["patient_id", "video_id", "actual_L", "actual_R"] + [m + "_" + s for m in NAMES for s in ("L", "R")])
            for i in range(len(pp)):
                writer.writerow([pp[i], vv[i], *yy[i], *[v for name in NAMES for v in predictions[name][i]]])
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for name in NAMES:
            per_person = {}
            for row in rows:
                error = statistics.mean(abs(float(row[name + "_" + side]) - float(row["actual_" + side])) for side in ("L", "R"))
                per_person.setdefault(row["patient_id"], []).append(error)
            h.require(abs(statistics.mean(statistics.mean(errors) for errors in per_person.values()) - expected[name]["person_mae"]) < 1e-10, "Independent kinematic CSV MAE mismatch")
    require_unchanged(before)
    require_unchanged(stage_before)
    comparisons = {name: {"validation_person_mae_reduction": val_results["global_mean"]["person_mae"] - val_results[name]["person_mae"],
                  "internal_oof_person_mae_reduction": oof_results["global_mean"]["person_mae"] - oof_results[name]["person_mae"],
                  "internal_folds_improved": sum(row["methods"][name]["person_mae"] < row["methods"]["global_mean"]["person_mae"] for row in folds)} for name in NAMES[1:]}
    result = {"status": "passed", "validation": val_results, "internal_training_oof": oof_results, "comparisons": comparisons,
              "folds": folds, "feature_records": 243, "feature_people": 212, "old_test_records_used": 0,
              "alpha": 1.0, "validation_performance_hyperparameter_search": False,
              "same_three_readiness_probes_reproduced": True, "saved_model_reproduction_passed": True,
              "independent_ridge_and_csv_checks": True, "source_files_unchanged": True,
              "physical_or_clinical_angle_validation": False, "elapsed_seconds": time.perf_counter() - started}
    h.save(folder / "kinematic_comparison_result.json", result)
    return {k: v for k, v in result.items() if k != "folds"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.session), ensure_ascii=True, indent=2))
    except Exception as error:
        import traceback
        h.save(ROOT / "control/desktop-sessions" / args.session / "kinematic_comparison_failure.json", {"error_type": type(error).__name__, "details": traceback.format_exc()})
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "details": "saved locally; no automatic retry"}))
        raise SystemExit(1)
