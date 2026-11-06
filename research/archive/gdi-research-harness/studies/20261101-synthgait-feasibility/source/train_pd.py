"""Train the retained PD4T LOSO classifiers from cached 1024-D features.

GaitXFormer::

    python train_pd.py --config configs/pd/default.yaml --log-dir PD4T/log_synthgait

Frozen V-JEPA2 baseline::

    python train_pd.py --config configs/pd/baseline.yaml --log-dir PD4T/log_baseline
"""

import argparse
import json
import os
import random
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pd4t import PD4TDataset, get_active_patient_ids, normalize_patient_ids
from utils.tools import get_config


def parse_args():
	parser = argparse.ArgumentParser(description="Train PD classifier with LOSO cross-validation")
	parser.add_argument(
		"--config",
		type=str,
		default="configs/pd/default.yaml",
		help="Path to YAML config",
	)
	parser.add_argument(
		"--log-dir",
		type=str,
		default="PD4T/log_synthgait",
		help="Directory for fold predictions and evaluation artifacts",
	)
	parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
	return parser.parse_args()


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


class MLPClassifier(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, num_classes),
		)

	def forward(self, x):
		return self.net(x)


def build_model(model_cfg, input_dim: int, num_classes: int) -> nn.Module:
	model_type = str(model_cfg.type).lower()
	if model_type == "linear":
		return nn.Linear(input_dim, num_classes)
	if model_type == "mlp":
		hidden_dim = int(model_cfg.hidden_dim)
		return MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
	raise ValueError(f"Unknown model type '{model_cfg.type}'. Expected one of: linear, mlp")


def validate_labels(
	labels: torch.Tensor,
	num_classes: int,
	context: str = "",
	sample_refs: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
	labels = labels.long()
	keep_mask = (labels >= 0) & (labels < num_classes)
	skip_mask = torch.zeros_like(keep_mask)

	# Backward-compatibility rule for legacy UPDRS annotations:
	# when training a 3-class model (labels 0/1/2), silently skip labels 3+.
	if int(num_classes) == 3:
		skip_mask = labels >= 3

	invalid_mask = ~(keep_mask | skip_mask)
	if not bool(invalid_mask.any()):
		return labels, keep_mask, int(skip_mask.sum().item())

	invalid_idx = torch.where(invalid_mask)[0].detach().cpu().tolist()
	invalid_vals = sorted(set(labels[invalid_mask].detach().cpu().tolist()))
	if sample_refs is not None:
		invalid_samples = [str(sample_refs[i]) for i in invalid_idx if 0 <= i < len(sample_refs)]
	else:
		invalid_samples = [f"batch_index={i}" for i in invalid_idx]

	prefix = f"{context} " if len(context) > 0 else ""
	raise ValueError(
		f"{prefix}Invalid label(s) found: {invalid_vals}. "
		f"Expected integer class ids in [0, {num_classes - 1}] (num_classes={num_classes}). "
		f"If num_classes=3, labels >= 3 are skipped automatically. "
		f"Invalid sample(s): {invalid_samples}. "
		"This usually means your annotation contains unsupported UPDRS labels or model.num_classes does not match the annotation label space."
	)


def preprocess_features(features: torch.Tensor) -> torch.Tensor:
	features = features.float()
	if features.ndim == 1:
		features = features.unsqueeze(0)
	elif features.ndim > 2:
		# Average temporal dimension if present, e.g., [B, T, C] -> [B, C]
		features = features.mean(dim=1)
	return features


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int):
	model.eval()
	all_preds = []
	all_targets = []
	skipped_total = 0

	for features, labels, sample_refs in tqdm(data_loader, desc="Evaluating", leave=False):
		features = preprocess_features(features).to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)
		labels, keep_mask, skipped_count = validate_labels(
			labels,
			num_classes=num_classes,
			context="[Eval]",
			sample_refs=sample_refs,
		)
		skipped_total += skipped_count
		if not bool(keep_mask.any()):
			continue

		features = features[keep_mask]
		labels = labels[keep_mask]

		logits = model(features)
		preds = torch.argmax(logits, dim=1)

		all_preds.extend(preds.detach().cpu().tolist())
		all_targets.extend(labels.detach().cpu().tolist())

	if skipped_total > 0:
		print(f"[Eval] Skipped {skipped_total} sample(s) with label >= 3 for num_classes=3.")

	return all_preds, all_targets


def confusion_matrix_from_lists(y_true, y_pred, num_classes: int):
	cm = np.zeros((num_classes, num_classes), dtype=np.int64)
	for t, p in zip(y_true, y_pred):
		if 0 <= t < num_classes and 0 <= p < num_classes:
			cm[t, p] += 1
	return cm


def macro_f1_from_confusion_matrix(cm: np.ndarray) -> float:
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


def accuracy_from_confusion_matrix(cm: np.ndarray) -> float:
	total = float(cm.sum())
	if total <= 0:
		return 0.0
	return float(np.trace(cm) / total)


def macro_recall_from_confusion_matrix(cm: np.ndarray) -> float:
	recalls = []
	for cls in range(cm.shape[0]):
		tp = float(cm[cls, cls])
		fn = float(cm[cls, :].sum() - tp)
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		recalls.append(recall)
	return float(np.mean(recalls))


def metrics_from_confusion_matrix(cm: np.ndarray) -> dict:
	return {
		"accuracy": accuracy_from_confusion_matrix(cm),
		"macro_recall": macro_recall_from_confusion_matrix(cm),
		"macro_f1": macro_f1_from_confusion_matrix(cm),
	}


def metrics_from_predictions(y_true, y_pred, num_classes: int) -> dict:
	cm = confusion_matrix_from_lists(y_true, y_pred, num_classes)
	return metrics_from_confusion_matrix(cm)


def mean_metric(metrics: list[dict], key: str) -> float:
	if len(metrics) == 0:
		return 0.0
	return float(np.mean([m[key] for m in metrics]))


def macro_f1_present_classes(y_true, cm: np.ndarray, num_classes: int) -> float:
	present_classes = sorted({int(y) for y in y_true if 0 <= int(y) < int(num_classes)})
	if len(present_classes) == 0:
		return 0.0

	f1_scores = []
	for cls in present_classes:
		tp = float(cm[cls, cls])
		fp = float(cm[:, cls].sum() - tp)
		fn = float(cm[cls, :].sum() - tp)

		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
		f1_scores.append(f1)

	return float(np.mean(f1_scores))


def plot_confusion_matrix(cm: np.ndarray, out_path: str):
	fig, ax = plt.subplots(figsize=(6, 6))
	im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
	fig.colorbar(im, ax=ax)

	ax.set_title("Confusion Matrix")
	ax.set_xlabel("Predicted label")
	ax.set_ylabel("True label")

	ticks = np.arange(cm.shape[0])
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)

	thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(
				j,
				i,
				str(cm[i, j]),
				ha="center",
				va="center",
				color="white" if cm[i, j] > thresh else "black",
			)

	fig.tight_layout()
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def save_fold_output(path: str, payload: dict):
	with open(path, "w") as f:
		json.dump(payload, f, indent=2)


def load_fold_output(path: str):
	with open(path, "r") as f:
		return json.load(f)


def get_excluded_patient_ids(cfg) -> list[str]:
	return normalize_patient_ids(cfg.dataset.exclude_patient_ids)


def get_val_only_patient_ids(cfg) -> list[str]:
	return normalize_patient_ids(cfg.dataset.val_only_patient_ids)


def get_score_column(cfg) -> str:
	return str(cfg.dataset.score_column)


def get_fixed_val_subjects_path(cfg) -> str | None:
	if not hasattr(cfg.dataset, "fixed_val_subjects_path"):
		return None
	path = getattr(cfg.dataset, "fixed_val_subjects_path")
	if path is None or str(path).strip() == "":
		return None
	return str(path)


def load_fixed_val_subjects(path: str | None) -> dict:
	if path is None:
		return {}
	with open(path, "r") as f:
		payload = json.load(f)

	fixed_val_subjects = {}
	for fold_id, fold_payload in payload.items():
		if isinstance(fold_payload, dict):
			val_subjects = fold_payload["val_subjects"]
		else:
			val_subjects = fold_payload
		fixed_val_subjects[int(fold_id)] = normalize_patient_ids(val_subjects)
	return fixed_val_subjects


def fixed_val_subjects_for_fold(fixed_val_subjects: dict, fold_id: int) -> list[str] | None:
	if len(fixed_val_subjects) == 0:
		return None
	if int(fold_id) not in fixed_val_subjects:
		raise KeyError(f"Missing fixed validation subjects for fold {fold_id}.")
	return fixed_val_subjects[int(fold_id)]


def cached_fold_is_compatible(
	payload: dict,
	expected_test_subject: str,
	excluded_patient_ids: list[str],
	val_only_patient_ids: list[str],
	active_patient_ids: list[str],
	fixed_val_subjects_path: str | None,
	score_column: str,
) -> bool:
	if payload.get("test_subject") != expected_test_subject:
		return False

	cached_excluded_patient_ids = payload.get("excluded_patient_ids")
	cached_val_only_patient_ids = payload.get("val_only_patient_ids")
	cached_active_patient_ids = payload.get("active_patient_ids")
	cached_fixed_val_subjects_path = payload.get("fixed_val_subjects_path")
	cached_score_column = payload.get("score_column")

	if len(excluded_patient_ids) > 0 and cached_excluded_patient_ids is None:
		return False
	if cached_excluded_patient_ids is not None and normalize_patient_ids(cached_excluded_patient_ids) != excluded_patient_ids:
		return False
	if len(val_only_patient_ids) > 0 and cached_val_only_patient_ids is None:
		return False
	if cached_val_only_patient_ids is not None and normalize_patient_ids(cached_val_only_patient_ids) != val_only_patient_ids:
		return False
	if cached_active_patient_ids is not None and normalize_patient_ids(cached_active_patient_ids) != active_patient_ids:
		return False
	if cached_fixed_val_subjects_path != fixed_val_subjects_path:
		return False
	if cached_score_column is None and score_column != "Old score ":
		return False
	if cached_score_column is not None and cached_score_column != score_column:
		return False

	return True


def train_one_fold(cfg, fold_id: int, device: torch.device, log_dir: str):
	excluded_patient_ids = get_excluded_patient_ids(cfg)
	val_only_patient_ids = get_val_only_patient_ids(cfg)
	active_patient_ids = get_active_patient_ids(excluded_patient_ids, val_only_patient_ids)
	expected_test_subject = active_patient_ids[int(fold_id)]
	fixed_val_subjects_path = get_fixed_val_subjects_path(cfg)
	fixed_val_subjects = load_fixed_val_subjects(fixed_val_subjects_path)
	fold_fixed_val_subjects = fixed_val_subjects_for_fold(fixed_val_subjects, int(fold_id))
	score_column = get_score_column(cfg)

	fold_txt_path = os.path.join(log_dir, f"fold_{fold_id:02d}_predictions.txt")
	if os.path.exists(fold_txt_path):
		payload = load_fold_output(fold_txt_path)
		if cached_fold_is_compatible(
			payload=payload,
			expected_test_subject=expected_test_subject,
			excluded_patient_ids=excluded_patient_ids,
			val_only_patient_ids=val_only_patient_ids,
			active_patient_ids=active_patient_ids,
			fixed_val_subjects_path=fixed_val_subjects_path,
			score_column=score_column,
		):
			predictions = payload["predictions"]
			targets = payload["targets"]
			metrics = metrics_from_predictions(targets, predictions, int(cfg.model.num_classes))
			print(f"[Fold {fold_id:02d}] Found cached results at {fold_txt_path}. Skipping training.")
			print(
				f"[Fold {fold_id:02d}] Cached test metrics -> "
				f"acc: {metrics['accuracy']:.4f}, "
				f"recall: {metrics['macro_recall']:.4f}, "
				f"f1: {metrics['macro_f1']:.4f}"
			)
			return predictions, targets, metrics
		print(f"[Fold {fold_id:02d}] Ignoring stale cached results at {fold_txt_path}; split config changed.")

	seed = int(cfg.train.seed) + int(fold_id)
	set_seed(seed)

	train_dataset = PD4TDataset(
		feat_dir=cfg.dataset.feat_dir,
		annot_path=cfg.dataset.annot_path,
		fold_id=fold_id,
		split="train",
		val_seed=int(cfg.train.seed),
		num_classes=int(cfg.model.num_classes),
		val_ratio=float(getattr(cfg.train, "val_ratio", 0.2)),
		min_val_subjects=int(getattr(cfg.train, "min_val_subjects", 3)),
		exclude_patient_ids=excluded_patient_ids,
		val_only_patient_ids=val_only_patient_ids,
		fixed_val_subjects=fold_fixed_val_subjects,
		score_column=score_column,
	)
	val_dataset = PD4TDataset(
		feat_dir=cfg.dataset.feat_dir,
		annot_path=cfg.dataset.annot_path,
		fold_id=fold_id,
		split="val",
		val_seed=int(cfg.train.seed),
		num_classes=int(cfg.model.num_classes),
		val_ratio=float(getattr(cfg.train, "val_ratio", 0.2)),
		min_val_subjects=int(getattr(cfg.train, "min_val_subjects", 3)),
		exclude_patient_ids=excluded_patient_ids,
		val_only_patient_ids=val_only_patient_ids,
		fixed_val_subjects=fold_fixed_val_subjects,
		score_column=score_column,
	)
	test_dataset = PD4TDataset(
		feat_dir=cfg.dataset.feat_dir,
		annot_path=cfg.dataset.annot_path,
		fold_id=fold_id,
		split="test",
		val_seed=int(cfg.train.seed),
		num_classes=int(cfg.model.num_classes),
		val_ratio=float(getattr(cfg.train, "val_ratio", 0.2)),
		min_val_subjects=int(getattr(cfg.train, "min_val_subjects", 3)),
		exclude_patient_ids=excluded_patient_ids,
		val_only_patient_ids=val_only_patient_ids,
		fixed_val_subjects=fold_fixed_val_subjects,
		score_column=score_column,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=int(cfg.train.batch_size),
		shuffle=True,
		num_workers=int(cfg.train.num_workers),
		pin_memory=True,
		drop_last=False,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=int(cfg.train.batch_size),
		shuffle=False,
		num_workers=int(cfg.train.num_workers),
		pin_memory=True,
		drop_last=False,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=int(cfg.train.batch_size),
		shuffle=False,
		num_workers=int(cfg.train.num_workers),
		pin_memory=True,
		drop_last=False,
	)

	print(
		f"[Fold {fold_id:02d}] split subjects -> "
		f"test: {test_dataset.test_subject}, val: {val_dataset.val_subjects}, "
		f"train_count: {len(train_dataset.train_subjects)}, "
		f"excluded: {excluded_patient_ids}, val_only: {val_only_patient_ids}"
	)

	model = build_model(
		model_cfg=cfg.model,
		input_dim=int(cfg.model.input_dim),
		num_classes=int(cfg.model.num_classes),
	).to(device)
	print(f"[Fold {fold_id:02d}] model.type = {cfg.model.type}")

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=float(cfg.train.lr),
		weight_decay=float(cfg.train.weight_decay),
	)

	best_macro_f1 = -1.0
	best_val_macro_f1 = -1.0
	best_metrics = {
		"accuracy": 0.0,
		"macro_recall": 0.0,
		"macro_f1": -1.0,
	}
	best_predictions = []
	best_targets = []

	epochs = int(cfg.train.epochs_per_fold)
	epoch_bar = tqdm(range(epochs), desc=f"Fold {fold_id:02d} epochs")
	for _ in epoch_bar:
		model.train()
		running_loss = 0.0
		n_samples = 0
		skipped_train_total = 0

		for features, labels, sample_refs in tqdm(train_loader, desc=f"Fold {fold_id:02d} train", leave=False):
			features = preprocess_features(features).to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			labels, keep_mask, skipped_count = validate_labels(
				labels,
				num_classes=int(cfg.model.num_classes),
				context=f"[Fold {fold_id:02d}][Train]",
				sample_refs=sample_refs,
			)
			skipped_train_total += skipped_count
			if not bool(keep_mask.any()):
				continue

			features = features[keep_mask]
			labels = labels[keep_mask]

			optimizer.zero_grad()
			logits = model(features)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

			batch_size = labels.size(0)
			running_loss += loss.item() * batch_size
			n_samples += batch_size

		if skipped_train_total > 0:
			print(
				f"[Fold {fold_id:02d}][Train] Skipped {skipped_train_total} sample(s) with label >= 3 for num_classes=3."
			)

		train_loss = running_loss / max(1, n_samples)

		val_preds, val_targets = evaluate(model, val_loader, device, int(cfg.model.num_classes))
		val_cm = confusion_matrix_from_lists(val_targets, val_preds, int(cfg.model.num_classes))
		val_macro_f1 = macro_f1_present_classes(
			y_true=val_targets,
			cm=val_cm,
			num_classes=int(cfg.model.num_classes),
		)

		if val_macro_f1 >= best_val_macro_f1:
			best_val_macro_f1 = val_macro_f1
			preds, targets = evaluate(model, test_loader, device, int(cfg.model.num_classes))
			cm = confusion_matrix_from_lists(targets, preds, int(cfg.model.num_classes))
			best_metrics = metrics_from_confusion_matrix(cm)
			best_macro_f1 = best_metrics["macro_f1"]
			best_predictions = preds
			best_targets = targets

		epoch_bar.set_postfix({
			"train_loss": f"{train_loss:.4f}",
			"best_val_f1": f"{best_val_macro_f1:.4f}",
			"best_test_acc": f"{best_metrics['accuracy']:.4f}",
			"best_test_recall": f"{best_metrics['macro_recall']:.4f}",
			"best_test_f1": f"{best_macro_f1:.4f}",
		})

	payload = {
		"fold_id": fold_id,
		"seed": seed,
		"test_subject": test_dataset.test_subject,
		"val_subjects": val_dataset.val_subjects,
		"excluded_patient_ids": excluded_patient_ids,
		"val_only_patient_ids": val_only_patient_ids,
		"active_patient_ids": active_patient_ids,
		"fixed_val_subjects_path": fixed_val_subjects_path,
		"score_column": score_column,
		"best_val_macro_f1": best_val_macro_f1,
		"best_accuracy": best_metrics["accuracy"],
		"best_macro_recall": best_metrics["macro_recall"],
		"best_macro_f1": best_macro_f1,
		"best_metrics": best_metrics,
		"predictions": best_predictions,
		"targets": best_targets,
	}
	save_fold_output(fold_txt_path, payload)
	print(
		f"[Fold {fold_id:02d}] Test metrics -> "
		f"acc: {best_metrics['accuracy']:.4f}, "
		f"recall: {best_metrics['macro_recall']:.4f}, "
		f"f1: {best_metrics['macro_f1']:.4f}"
	)
	print(f"[Fold {fold_id:02d}] Saved fold output to {fold_txt_path}")
	return best_predictions, best_targets, best_metrics


def main():
	opts = parse_args()
	cfg = get_config(opts.config)

	if opts.seed is not None:
		cfg.train.seed = int(opts.seed)

	os.makedirs(opts.log_dir, exist_ok=True)

	set_seed(int(cfg.train.seed))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	excluded_patient_ids = get_excluded_patient_ids(cfg)
	val_only_patient_ids = get_val_only_patient_ids(cfg)
	active_patient_ids = get_active_patient_ids(excluded_patient_ids, val_only_patient_ids)
	fixed_val_subjects_path = get_fixed_val_subjects_path(cfg)
	score_column = get_score_column(cfg)
	configured_num_folds = int(getattr(cfg.train, "num_folds", len(active_patient_ids)))
	num_folds = len(active_patient_ids)
	if configured_num_folds != num_folds:
		print(
			f"[INFO] train.num_folds={configured_num_folds}, but {num_folds} active patient(s) remain "
			f"after exclusions. Running {num_folds} LOSO folds."
		)
	print(f"Active patient ids ({len(active_patient_ids)}): {active_patient_ids}")
	print(f"Excluded patient ids ({len(excluded_patient_ids)}): {excluded_patient_ids}")
	print(f"Validation-only patient ids ({len(val_only_patient_ids)}): {val_only_patient_ids}")
	print(f"Fixed validation subjects path: {fixed_val_subjects_path}")
	print(f"Score column: {score_column}")

	all_predictions = []
	all_targets = []
	per_fold_metrics = []

	for fold_id in tqdm(range(num_folds), desc="LOSO folds"):
		fold_preds, fold_targets, fold_metrics = train_one_fold(
			cfg=cfg,
			fold_id=fold_id,
			device=device,
			log_dir=opts.log_dir,
		)
		all_predictions.extend(fold_preds)
		all_targets.extend(fold_targets)
		per_fold_metrics.append(fold_metrics)

	cm = confusion_matrix_from_lists(all_targets, all_predictions, int(cfg.model.num_classes))
	overall_metrics = metrics_from_confusion_matrix(cm)

	cm_path = os.path.join(opts.log_dir, "confusion_matrix.png")
	plot_confusion_matrix(cm, cm_path)

	summary = {
		"num_folds": num_folds,
		"configured_num_folds": configured_num_folds,
		"excluded_patient_ids": excluded_patient_ids,
		"val_only_patient_ids": val_only_patient_ids,
		"active_patient_ids": active_patient_ids,
		"fixed_val_subjects_path": fixed_val_subjects_path,
		"score_column": score_column,
		"overall_accuracy": overall_metrics["accuracy"],
		"overall_macro_recall": overall_metrics["macro_recall"],
		"overall_macro_f1": overall_metrics["macro_f1"],
		"mean_best_fold_accuracy": mean_metric(per_fold_metrics, "accuracy"),
		"mean_best_fold_macro_recall": mean_metric(per_fold_metrics, "macro_recall"),
		"mean_best_fold_macro_f1": mean_metric(per_fold_metrics, "macro_f1"),
		"fold_metrics": per_fold_metrics,
		"confusion_matrix": cm.tolist(),
	}

	summary_path = os.path.join(opts.log_dir, "summary.txt")
	with open(summary_path, "w") as f:
		json.dump(summary, f, indent=2)

	print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
	print(f"Overall Macro-Recall: {overall_metrics['macro_recall']:.4f}")
	print(f"Overall Macro-F1: {overall_metrics['macro_f1']:.4f}")
	print(f"Confusion matrix saved to: {cm_path}")
	print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
	main()
