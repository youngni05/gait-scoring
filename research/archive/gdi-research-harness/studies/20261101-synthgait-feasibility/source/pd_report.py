"""Compare retained GaitXFormer and V-JEPA2 PD4T LOSO fold outputs.

Run ``python pd_report.py`` after both canonical log directories exist.
"""

import argparse
import glob
import json
import os
import re

import numpy as np


def parse_args():
	parser = argparse.ArgumentParser(description="Report PD metrics from saved LOSO fold outputs")
	parser.add_argument(
		"--log-dir",
		type=str,
		default="PD4T/log_synthgait",
		help="Directory containing fold_<id>_predictions.txt files",
	)
	parser.add_argument(
		"--log-dir-baseline",
		type=str,
		default="PD4T/log_baseline",
		help="Baseline directory containing fold_<id>_predictions.txt files",
	)
	return parser.parse_args()


def load_fold_output(path: str):
	with open(path, "r") as f:
		return json.load(f)


def parse_fold_id_from_path(path: str):
	match = re.search(r"fold_(\d+)_predictions\.txt$", os.path.basename(path))
	if match is None:
		return None
	return int(match.group(1))


def collect_fold_payloads(log_dir: str):
	fold_paths = sorted(glob.glob(os.path.join(log_dir, "fold_*_predictions.txt")))
	if len(fold_paths) == 0:
		raise FileNotFoundError(
			f"No fold prediction files found in '{log_dir}'. "
			"Expected files like fold_00_predictions.txt"
		)

	folds = {}
	for path in fold_paths:
		payload = load_fold_output(path)
		fold_id = payload.get("fold_id", None)
		if fold_id is None:
			fold_id = parse_fold_id_from_path(path)
		if fold_id is None:
			raise ValueError(f"Could not infer fold_id for file: {path}")
		fold_id = int(fold_id)

		preds = payload.get("predictions", [])
		targets = payload.get("targets", [])
		if len(preds) != len(targets):
			raise ValueError(
				f"Mismatched lengths in {path}: "
				f"len(predictions)={len(preds)} vs len(targets)={len(targets)}"
			)

		if fold_id in folds:
			raise ValueError(f"Duplicate fold_id={fold_id} found in {log_dir}")

		folds[fold_id] = {
			"path": path,
			"payload": payload,
		}

	return folds


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
	"""Equivalent to scipy.stats.rankdata(values, method='average')."""
	values = np.asarray(values)
	order = np.argsort(values, kind="mergesort")
	sorted_vals = values[order]
	ranks = np.empty(len(values), dtype=np.float64)

	i = 0
	while i < len(values):
		j = i + 1
		while j < len(values) and sorted_vals[j] == sorted_vals[i]:
			j += 1
		# 1-based average rank for ties in [i, j)
		avg_rank = (i + 1 + j) / 2.0
		ranks[order[i:j]] = avg_rank
		i = j

	return ranks


def spearman_correlation(y_true, y_pred) -> float:
	y_true = np.asarray(y_true, dtype=np.float64)
	y_pred = np.asarray(y_pred, dtype=np.float64)
	if y_true.size == 0:
		return 0.0
	if y_true.size != y_pred.size:
		raise ValueError("y_true and y_pred must have the same length for Spearman correlation.")

	r_true = rankdata_average_ties(y_true)
	r_pred = rankdata_average_ties(y_pred)

	r_true_mean = r_true.mean()
	r_pred_mean = r_pred.mean()
	den = np.sqrt(np.sum((r_true - r_true_mean) ** 2) * np.sum((r_pred - r_pred_mean) ** 2))
	if den == 0.0:
		return 0.0
	return float(np.sum((r_true - r_true_mean) * (r_pred - r_pred_mean)) / den)


def quadratic_weighted_kappa(y_true, y_pred) -> float:
	y_true = np.asarray(y_true, dtype=np.int64)
	y_pred = np.asarray(y_pred, dtype=np.int64)
	if y_true.size == 0:
		return 0.0
	if y_true.size != y_pred.size:
		raise ValueError("y_true and y_pred must have the same length for QWK.")

	min_rating = int(min(y_true.min(), y_pred.min()))
	max_rating = int(max(y_true.max(), y_pred.max()))
	n_ratings = max_rating - min_rating + 1

	if n_ratings <= 1:
		return 1.0

	# Observed matrix O
	o = np.zeros((n_ratings, n_ratings), dtype=np.float64)
	for a, b in zip(y_true, y_pred):
		o[a - min_rating, b - min_rating] += 1.0

	# Expected matrix E
	hist_true = np.bincount(y_true - min_rating, minlength=n_ratings).astype(np.float64)
	hist_pred = np.bincount(y_pred - min_rating, minlength=n_ratings).astype(np.float64)
	e = np.outer(hist_true, hist_pred) / float(y_true.size)

	# Quadratic weight matrix W
	denom = float((n_ratings - 1) ** 2)
	w = np.zeros((n_ratings, n_ratings), dtype=np.float64)
	for i in range(n_ratings):
		for j in range(n_ratings):
			w[i, j] = ((i - j) ** 2) / denom

	num = float(np.sum(w * o))
	den = float(np.sum(w * e))
	if den == 0.0:
		return 0.0
	return 1.0 - (num / den)


def mae_score(y_true, y_pred) -> float:
	y_true = np.asarray(y_true, dtype=np.float64)
	y_pred = np.asarray(y_pred, dtype=np.float64)
	if y_true.size == 0:
		return 0.0
	if y_true.size != y_pred.size:
		raise ValueError("y_true and y_pred must have the same length for MAE.")
	return float(np.mean(np.abs(y_true - y_pred)))


def macro_f1_score(y_true, y_pred) -> float:
	y_true = np.asarray(y_true, dtype=np.int64)
	y_pred = np.asarray(y_pred, dtype=np.int64)
	if y_true.size == 0:
		return 0.0
	if y_true.size != y_pred.size:
		raise ValueError("y_true and y_pred must have the same length for Macro-F1.")

	labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
	if len(labels) == 0:
		return 0.0

	label_to_index = {label: idx for idx, label in enumerate(labels)}
	cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
	for t, p in zip(y_true, y_pred):
		cm[label_to_index[int(t)], label_to_index[int(p)]] += 1

	f1_scores = []
	for cls in range(cm.shape[0]):
		tp = float(cm[cls, cls])
		fp = float(cm[:, cls].sum() - tp)
		fn = float(cm[cls, :].sum() - tp)
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
		f1_scores.append(f1)

	return float(np.mean(f1_scores))


def compute_metrics(y_true, y_pred):
	return {
		"macro_f1": macro_f1_score(y_true, y_pred),
		"qwk": quadratic_weighted_kappa(y_true, y_pred),
		"spearman": spearman_correlation(y_true, y_pred),
		"mae": mae_score(y_true, y_pred),
	}


def main():
	opts = parse_args()

	main_folds = collect_fold_payloads(opts.log_dir)
	baseline_folds = collect_fold_payloads(opts.log_dir_baseline)

	main_ids = set(main_folds.keys())
	baseline_ids = set(baseline_folds.keys())
	common_ids = sorted(main_ids & baseline_ids)
	missing_in_baseline = sorted(main_ids - baseline_ids)
	missing_in_main = sorted(baseline_ids - main_ids)

	if len(common_ids) == 0:
		raise ValueError(
			"No common fold ids between --log-dir and --log-dir-baseline. "
			f"main={sorted(main_ids)}, baseline={sorted(baseline_ids)}"
		)

	if len(missing_in_baseline) > 0 or len(missing_in_main) > 0:
		raise ValueError(
			"Fold id mismatch between compared directories. "
			f"Missing in baseline: {missing_in_baseline}; Missing in main: {missing_in_main}"
		)

	metric_names = ["macro_f1", "qwk", "spearman", "mae"]
	per_fold_results = []
	diff_values = {name: [] for name in metric_names}

	for fold_id in common_ids:
		main_payload = main_folds[fold_id]["payload"]
		base_payload = baseline_folds[fold_id]["payload"]

		main_targets = main_payload.get("targets", [])
		main_preds = main_payload.get("predictions", [])
		base_targets = base_payload.get("targets", [])
		base_preds = base_payload.get("predictions", [])

		main_metrics = compute_metrics(main_targets, main_preds)
		base_metrics = compute_metrics(base_targets, base_preds)

		diff_metrics = {}
		for name in metric_names:
			d = float(main_metrics[name] - base_metrics[name])
			diff_metrics[name] = d
			diff_values[name].append(d)

		per_fold_results.append(
			{
				"fold_id": int(fold_id),
				"num_samples_main": len(main_targets),
				"num_samples_baseline": len(base_targets),
				"main": main_metrics,
				"baseline": base_metrics,
				"diff_main_minus_baseline": diff_metrics,
			}
		)

	diff_summary = {}
	for name in metric_names:
		arr = np.asarray(diff_values[name], dtype=np.float64)
		diff_summary[name] = {
			"mean": float(arr.mean()) if arr.size > 0 else 0.0,
			"std": float(arr.std()) if arr.size > 0 else 0.0,
		}

	report = {
		"log_dir": opts.log_dir,
		"log_dir_baseline": opts.log_dir_baseline,
		"num_compared_folds": len(common_ids),
		"per_fold": per_fold_results,
		"diff_summary_main_minus_baseline": diff_summary,
	}

	report_path = os.path.join(opts.log_dir, "pd_compare_report.txt")
	with open(report_path, "w") as f:
		json.dump(report, f, indent=2)

	print(f"Main log dir: {opts.log_dir}")
	print(f"Baseline log dir: {opts.log_dir_baseline}")
	print(f"Compared folds: {len(common_ids)}")
	print("Diff summary (main - baseline):")
	for name in metric_names:
		print(f"  {name}: mean={diff_summary[name]['mean']:.6f}, std={diff_summary[name]['std']:.6f}")
	print(f"Saved compare report to: {report_path}")


if __name__ == "__main__":
	main()
