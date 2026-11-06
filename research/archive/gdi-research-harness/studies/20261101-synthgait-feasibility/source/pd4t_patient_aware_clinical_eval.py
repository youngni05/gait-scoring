#!/usr/bin/env python3
"""
Patient-aware clinical-anchor evaluation for GaitXFormer on PD4T.

This script is meant for PD4T videos without MoCap ground truth. It evaluates
whether predicted gait features align with a clinical score/UPDRS anchor while
avoiding the mistake of treating many videos from the same patient as
independent samples.

Main outputs:
  1. raw_predictions.csv
  2. video_level_anchor_validity_descriptive.csv
  3. patient_score_predictions.csv
  4. patient_score_anchor_validity_clustered.csv
  5. within_patient_pairs.csv
  6. within_patient_directional_validity_clustered.csv
  7. patient_aware_summary.csv

Typical use with model inference:
  python pd4t_patient_aware_clinical_eval.py \
      --config configs/eval/pd4t_patient_aware_clinical_eval.yaml \
      --checkpoint "$GAITXFORMER_CHECKPOINT_FILE" \
      --out-dir pd4t_patient_aware_outputs

Typical use if raw_predictions.csv already exists:
  python pd4t_patient_aware_clinical_eval.py \
      --config configs/eval/pd4t_patient_aware_clinical_eval.yaml \
      --predictions /path/to/raw_predictions.csv \
      --out-dir pd4t_patient_aware_outputs
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.tools import get_config, resolve_checkpoint_path
from data import pd4t as pd4t_module
from model.gaitxformer import GaitXFormer

PD4TVideoDataset = pd4t_module.PD4TVideoDataset

try:
    DEFAULT_PATIENT_IDS = list(pd4t_module.PATIENT_IDS)
except Exception:
    DEFAULT_PATIENT_IDS = None

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) if isinstance(v, dict) else v for k, v in obj.items()}
    if hasattr(obj, "items"):
        return {k: to_plain_dict(v) if hasattr(v, "items") else v for k, v in obj.items()}
    raise TypeError(f"Cannot convert object of type {type(obj)} to dict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-aware PD4T clinical-anchor evaluation for GaitXFormer."
    )
    parser.add_argument("--config", required=True, type=str, help="YAML config path.")
    parser.add_argument("--checkpoint", default=None, type=str, help="Model checkpoint path. Required unless --predictions is provided.")
    parser.add_argument("--predictions", default=None, type=Path, help="Existing raw_predictions.csv. If provided, skips model inference.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--device", default=None, type=str, help="Override device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Model / inference helpers
# -----------------------------------------------------------------------------

def unpack_batch(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (videos, valids) from a dataset batch.

    Supported formats:
      - dict with video and valid-frame keys
      - tuple/list where batch[0] is videos and batch[1] is valids
    """
    if isinstance(batch, dict):
        videos = None
        for key in ("videos", "video", "frames", "inputs", "x"):
            if key in batch:
                videos = batch[key]
                break
        valids = None
        for key in ("valids", "valid", "valid_frames", "mask"):
            if key in batch:
                valids = batch[key]
                break
        if videos is None:
            raise KeyError(f"Could not find video tensor in batch dict keys: {list(batch.keys())}")
        return videos, valids

    if isinstance(batch, (tuple, list)):
        if len(batch) < 1:
            raise ValueError("Empty batch tuple/list")
        videos = batch[0]
        valids = batch[1] if len(batch) > 1 else None
        return videos, valids

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def infer_valids_from_videos(videos: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Fallback valid-frame mask if dataset does not return valids.

    Returning valids from the dataset is preferred. This fallback assumes padded
    frames have near-zero energy before/after normalization.
    """
    if videos.ndim != 5:
        raise ValueError(f"Expected 5D videos tensor, got shape {tuple(videos.shape)}")

    if videos.shape[1] == n_frames:  # B,T,C,H,W
        energy = videos.detach().abs().sum(dim=(2, 3, 4))
    elif videos.shape[2] == n_frames:  # B,C,T,H,W
        energy = videos.detach().abs().sum(dim=(1, 3, 4))
    else:
        energy = videos.detach().abs().sum(dim=tuple(range(2, videos.ndim)))

    return energy.gt(0)


def get_output_tensor(outputs: Any) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        for key in ("pred", "preds", "prediction", "predictions", "outputs", "logits"):
            if key in outputs and torch.is_tensor(outputs[key]):
                return outputs[key]
        raise KeyError(f"Could not find prediction tensor in output dict keys: {list(outputs.keys())}")
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def load_model(args: Any, checkpoint_path: str, device: torch.device) -> GaitXFormer:
    gait_features = list(cfg_get(cfg_get(args, "data"), "gait_features"))
    model_kwargs = to_plain_dict(cfg_get(args, "model"))
    model = GaitXFormer(**model_kwargs, n_out=len(gait_features))
    model.to(device)

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def build_dataset_for_patient(args: Any, patient_id: str) -> PD4TVideoDataset:
    data_cfg = cfg_get(args, "data")
    gait_features = list(cfg_get(data_cfg, "gait_features"))

    dataset_kwargs = {
        "video_dir": cfg_get(data_cfg, "video_dir", cfg_get(data_cfg, "rgb_dir")),
        "annot_path": cfg_get(data_cfg, "annot_path", cfg_get(data_cfg, "annot_dir")),
        "patient_id": str(patient_id).zfill(3),
        "height": cfg_get(data_cfg, "height", 256),
        "width": cfg_get(data_cfg, "width", None),
        "n_frames": cfg_get(data_cfg, "n_frames", 32),
        "target_fps": cfg_get(data_cfg, "target_fps", 16),
        "clip_seconds": cfg_get(data_cfg, "clip_seconds", 5),
        "gait_stats_path": cfg_get(data_cfg, "gait_stats_path", None),
        "gait_features": gait_features,
    }

    if dataset_kwargs["video_dir"] is None:
        raise ValueError("data.video_dir or data.rgb_dir is required for PD4T evaluation")
    if dataset_kwargs["annot_path"] is None:
        raise ValueError("data.annot_path or data.annot_dir is required for PD4T evaluation")

    return PD4TVideoDataset(**dataset_kwargs)


def get_patient_ids(args: Any) -> List[str]:
    data_cfg = cfg_get(args, "data")
    analysis_cfg = cfg_get(args, "analysis")
    patient_ids = cfg_get(data_cfg, "patient_ids", None)
    if patient_ids is None:
        patient_ids = cfg_get(analysis_cfg, "patient_ids", None)
    if patient_ids is None:
        if DEFAULT_PATIENT_IDS is None:
            raise ValueError("No patient_ids in config and data.pd4t.PATIENT_IDS is unavailable")
        patient_ids = DEFAULT_PATIENT_IDS
    return [str(pid).zfill(3) for pid in patient_ids]


def video_metadata(dataset: PD4TVideoDataset, local_index: int) -> Dict[str, Any]:
    video_path = dataset.video_files[int(local_index)]
    video_name = os.path.basename(video_path)
    annotation_id = dataset._annotation_id_from_video_path(video_path)
    score = dataset._score_from_video_path(video_path)
    return {
        "patient_id": str(dataset.patient_id).zfill(3),
        "video_path": video_path,
        "video_name": video_name,
        "annotation_id": annotation_id,
        "updrs_score": int(score),
    }


@torch.no_grad()
def run_inference(args: Any, opts: argparse.Namespace) -> pd.DataFrame:
    if opts.checkpoint is None:
        raise ValueError("--checkpoint is required when --predictions is not provided")

    device_str = opts.device or cfg_get(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    data_cfg = cfg_get(args, "data")
    eval_cfg = cfg_get(args, "eval")
    gait_features = list(cfg_get(data_cfg, "gait_features"))
    batch_size = int(cfg_get(eval_cfg, "batch_size", 8))
    num_workers = int(cfg_get(eval_cfg, "num_workers", 4))
    pin_memory = bool(cfg_get(eval_cfg, "pin_memory", True))
    min_valid_frames = int(cfg_get(args, "min_valid_frames", 10))
    denormalize = bool(cfg_get(eval_cfg, "denormalize", True))

    model = load_model(args, opts.checkpoint, device)
    patient_ids = get_patient_ids(args)

    rows: List[Dict[str, Any]] = []
    total_videos = 0
    evaluated_videos = 0
    skipped_short_videos = 0

    for patient_id in tqdm(patient_ids, desc="patients"):
        dataset = build_dataset_for_patient(args, patient_id)
        if len(dataset) == 0:
            print(f"[warning] patient {patient_id}: no annotated videos found")
            continue

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        running_dataset_idx = 0
        for batch in tqdm(loader, desc=f"patient {patient_id}", leave=False):
            videos, valids = unpack_batch(batch)
            batch_total = int(videos.shape[0])
            batch_indices = torch.arange(running_dataset_idx, running_dataset_idx + batch_total, dtype=torch.long)
            running_dataset_idx += batch_total

            if valids is None:
                n_frames = int(cfg_get(data_cfg, "n_frames", 32))
                valids = infer_valids_from_videos(videos, n_frames=n_frames)

            valid_counts = valids.sum(dim=1)
            mask = valid_counts >= min_valid_frames

            total_videos += batch_total
            evaluated_videos += int(mask.sum().item())
            skipped_short_videos += int((~mask).sum().item())

            if int(mask.sum().item()) == 0:
                continue

            selected_indices = batch_indices[mask.cpu()].tolist()
            videos = videos[mask].to(device, non_blocking=True)
            valids = valids[mask].to(device, non_blocking=True)
            valid_counts_sel = valid_counts[mask.cpu()].cpu().numpy()

            outputs = model(videos, valids)
            preds = get_output_tensor(outputs)

            if denormalize:
                preds = dataset.denormalize_gait_features(preds)

            preds_np = preds.detach().cpu().numpy()

            for row_i, local_idx in enumerate(selected_indices):
                row = video_metadata(dataset, local_idx)
                row["valid_count"] = int(valid_counts_sel[row_i])
                for feat_i, feat_name in enumerate(gait_features):
                    row[feat_name] = float(preds_np[row_i, feat_i])
                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"total_videos={total_videos}, evaluated_videos={evaluated_videos}, skipped_short_videos={skipped_short_videos}")
    return df


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------

def finite_xy(x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def safe_corr(x: Sequence[float], y: Sequence[float], method: str = "spearman") -> float:
    x, y = finite_xy(x, y)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if scipy_stats is not None:
        if method == "spearman":
            return float(scipy_stats.spearmanr(x, y).correlation)
        return float(scipy_stats.pearsonr(x, y).statistic)
    return float(np.corrcoef(x, y)[0, 1])


def safe_corr_p(x: Sequence[float], y: Sequence[float], method: str = "spearman") -> float:
    x, y = finite_xy(x, y)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0 or scipy_stats is None:
        return np.nan
    if method == "spearman":
        return float(scipy_stats.spearmanr(x, y).pvalue)
    return float(scipy_stats.pearsonr(x, y).pvalue)


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    cluster_col: str,
    metric_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 2000,
    seed: int = 123,
) -> Tuple[float, float]:
    if df.empty or cluster_col not in df.columns:
        return np.nan, np.nan

    clusters = np.array(sorted(df[cluster_col].dropna().astype(str).unique()))
    if clusters.size < 2:
        return np.nan, np.nan

    groups = {cluster: df[df[cluster_col].astype(str) == cluster] for cluster in clusters}
    rng = np.random.default_rng(seed)
    vals: List[float] = []

    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=clusters.size, replace=True)
        boot_df = pd.concat([groups[c] for c in sampled], ignore_index=True)
        val = metric_fn(boot_df)
        if np.isfinite(val):
            vals.append(float(val))

    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def direction_to_sign(direction: Optional[str]) -> Optional[int]:
    if direction is None:
        return None
    d = str(direction).strip().lower()
    if d in ("increase", "+", "+1", "positive", "higher"):
        return +1
    if d in ("decrease", "-", "-1", "negative", "lower"):
        return -1
    if d in ("unknown", "none", "na", "n/a", "skip", "0"):
        return None
    raise ValueError(f"Unknown expected direction: {direction}")


def sign_to_text(sign: Optional[int]) -> str:
    if sign == +1:
        return "increase"
    if sign == -1:
        return "decrease"
    return "unknown"


def get_expected_directions(args: Any, gait_features: Sequence[str]) -> Dict[str, Optional[int]]:
    raw = cfg_get(cfg_get(args, "analysis"), "expected_directions", {}) or {}
    return {feat: direction_to_sign(cfg_get(raw, feat, "unknown")) for feat in gait_features}


def metric_srm(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    sd = np.std(values, ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return np.nan
    return float(np.mean(values) / sd)


def fmt_float(x: Any, digits: int = 3) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        return ""
    return f"{xf:.{digits}g}"


# -----------------------------------------------------------------------------
# Patient-aware analyses
# -----------------------------------------------------------------------------

def video_level_anchor_validity_descriptive(
    df: pd.DataFrame,
    gait_features: Sequence[str],
    expected: Dict[str, Optional[int]],
) -> pd.DataFrame:
    """Video-level descriptive correlations.

    These p-values treat videos as independent, so use them only descriptively.
    """
    rows = []
    for feat in gait_features:
        use = df[["patient_id", "updrs_score", feat]].dropna()
        pearson = safe_corr(use[feat], use["updrs_score"], "pearson")
        spearman = safe_corr(use[feat], use["updrs_score"], "spearman")
        sign = expected.get(feat)
        rows.append({
            "parameter": feat,
            "n_videos": int(len(use)),
            "n_patients": int(use["patient_id"].nunique()),
            "expected_direction_vs_higher_score": sign_to_text(sign),
            "pearson_r_video_level_descriptive": pearson,
            "pearson_p_naive_video_level": safe_corr_p(use[feat], use["updrs_score"], "pearson"),
            "spearman_r_video_level_descriptive": spearman,
            "spearman_p_naive_video_level": safe_corr_p(use[feat], use["updrs_score"], "spearman"),
            "direction_match": bool(np.sign(spearman) == sign) if sign is not None and np.isfinite(spearman) else np.nan,
        })
    return pd.DataFrame(rows)


def aggregate_patient_score(df: pd.DataFrame, gait_features: Sequence[str]) -> pd.DataFrame:
    """Average videos within each patient and clinical-score level."""
    agg = (
        df.groupby(["patient_id", "updrs_score"], as_index=False)
        .agg(
            n_videos=("video_name", "count"),
            **{feat: (feat, "mean") for feat in gait_features},
        )
    )
    return agg


def patient_score_anchor_validity_clustered(
    ps_df: pd.DataFrame,
    gait_features: Sequence[str],
    expected: Dict[str, Optional[int]],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Clinical-anchor correlations on patient-score aggregated rows.

    Uncertainty is estimated by patient-level cluster bootstrap.
    """
    rows = []
    for feat_i, feat in enumerate(gait_features):
        use = ps_df[["patient_id", "updrs_score", feat]].dropna().copy()
        sign = expected.get(feat)

        def spearman_metric(d: pd.DataFrame) -> float:
            return safe_corr(d[feat], d["updrs_score"], "spearman")

        def pearson_metric(d: pd.DataFrame) -> float:
            return safe_corr(d[feat], d["updrs_score"], "pearson")

        spearman = spearman_metric(use)
        pearson = pearson_metric(use)
        sp_lo, sp_hi = cluster_bootstrap_ci(use, "patient_id", spearman_metric, n_boot=n_boot, seed=seed + 1000 * feat_i + 1)
        pe_lo, pe_hi = cluster_bootstrap_ci(use, "patient_id", pearson_metric, n_boot=n_boot, seed=seed + 1000 * feat_i + 2)

        rows.append({
            "parameter": feat,
            "n_patient_score_rows": int(len(use)),
            "n_patients": int(use["patient_id"].nunique()),
            "expected_direction_vs_higher_score": sign_to_text(sign),
            "spearman_r_patient_score": spearman,
            "spearman_ci95_low_clustered": sp_lo,
            "spearman_ci95_high_clustered": sp_hi,
            "pearson_r_patient_score": pearson,
            "pearson_ci95_low_clustered": pe_lo,
            "pearson_ci95_high_clustered": pe_hi,
            "direction_match": bool(np.sign(spearman) == sign) if sign is not None and np.isfinite(spearman) else np.nan,
        })
    return pd.DataFrame(rows)


def make_within_patient_pairs(
    ps_df: pd.DataFrame,
    gait_features: Sequence[str],
    pair_strategy: str = "adjacent",
) -> pd.DataFrame:
    """Create pairs within each patient, oriented from lower score to higher score.

    Input should already be patient-score aggregated.
    """
    pair_rows = []
    for patient_id, g in ps_df.groupby("patient_id"):
        g = g.sort_values("updrs_score").reset_index(drop=True)
        if len(g) < 2 or g["updrs_score"].nunique() < 2:
            continue

        if pair_strategy == "all":
            index_pairs = list(itertools.combinations(range(len(g)), 2))
        elif pair_strategy == "adjacent":
            index_pairs = [(i, i + 1) for i in range(len(g) - 1)]
        else:
            raise ValueError("analysis.pair_strategy must be 'adjacent' or 'all'")

        for low_i, high_i in index_pairs:
            low = g.iloc[low_i]
            high = g.iloc[high_i]
            delta_score = float(high["updrs_score"] - low["updrs_score"])
            if delta_score <= 0:
                continue

            row = {
                "patient_id": str(patient_id),
                "score_low": float(low["updrs_score"]),
                "score_high": float(high["updrs_score"]),
                "delta_score": delta_score,
                "n_videos_low": int(low.get("n_videos", 0)),
                "n_videos_high": int(high.get("n_videos", 0)),
            }
            for feat in gait_features:
                row[f"low_{feat}"] = float(low[feat])
                row[f"high_{feat}"] = float(high[feat])
                row[f"delta_{feat}"] = float(high[feat] - low[feat])
                row[f"delta_per_score_{feat}"] = float((high[feat] - low[feat]) / delta_score)
            pair_rows.append(row)
    return pd.DataFrame(pair_rows)


def within_patient_directional_validity_clustered(
    pairs: pd.DataFrame,
    gait_features: Sequence[str],
    expected: Dict[str, Optional[int]],
    n_boot: int,
    seed: int,
    normalize_delta_by_score: bool = True,
) -> pd.DataFrame:
    """Within-patient severity-direction analysis with patient cluster bootstrap."""
    rows = []
    suffix = "delta_per_score" if normalize_delta_by_score else "delta"

    for feat_i, feat in enumerate(gait_features):
        sign = expected.get(feat)
        delta_col = f"{suffix}_{feat}"
        raw_delta_col = f"delta_{feat}"
        if delta_col not in pairs.columns:
            rows.append({"parameter": feat, "n_pairs": 0})
            continue

        use_cols = ["patient_id", "delta_score", delta_col, raw_delta_col]
        use = pairs[use_cols].dropna().copy()

        if sign is not None:
            use["aligned_change"] = sign * use[delta_col].astype(float)
            use["matches_expected_direction"] = use["aligned_change"] > 0
        else:
            use["aligned_change"] = np.nan
            use["matches_expected_direction"] = np.nan

        def pct_expected_metric(d: pd.DataFrame) -> float:
            vals = d["matches_expected_direction"].dropna().astype(float)
            if len(vals) == 0:
                return np.nan
            return float(vals.mean() * 100.0)

        def mean_aligned_metric(d: pd.DataFrame) -> float:
            vals = d["aligned_change"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return np.nan
            return float(np.mean(vals))

        def srm_metric(d: pd.DataFrame) -> float:
            vals = d["aligned_change"].to_numpy(dtype=float)
            return metric_srm(vals)

        pct_expected = pct_expected_metric(use)
        mean_aligned = mean_aligned_metric(use)
        srm = srm_metric(use)

        pct_lo, pct_hi = cluster_bootstrap_ci(use, "patient_id", pct_expected_metric, n_boot=n_boot, seed=seed + 1000 * feat_i + 101)
        mean_lo, mean_hi = cluster_bootstrap_ci(use, "patient_id", mean_aligned_metric, n_boot=n_boot, seed=seed + 1000 * feat_i + 102)
        srm_lo, srm_hi = cluster_bootstrap_ci(use, "patient_id", srm_metric, n_boot=n_boot, seed=seed + 1000 * feat_i + 103)

        # Also report raw delta vs delta score correlation inside pairs. This is descriptive.
        pair_spearman = safe_corr(use[raw_delta_col], use["delta_score"], "spearman")

        rows.append({
            "parameter": feat,
            "n_pairs": int(len(use)),
            "n_patients_with_pairs": int(use["patient_id"].nunique()),
            "expected_direction_with_higher_score": sign_to_text(sign),
            "delta_normalized_by_score": bool(normalize_delta_by_score),
            "percent_pairs_expected_direction": pct_expected,
            "percent_pairs_expected_direction_ci95_low_clustered": pct_lo,
            "percent_pairs_expected_direction_ci95_high_clustered": pct_hi,
            "mean_severity_aligned_change": mean_aligned,
            "mean_severity_aligned_change_ci95_low_clustered": mean_lo,
            "mean_severity_aligned_change_ci95_high_clustered": mean_hi,
            "severity_aligned_effect_size_srm": srm,
            "severity_aligned_effect_size_srm_ci95_low_clustered": srm_lo,
            "severity_aligned_effect_size_srm_ci95_high_clustered": srm_hi,
            "spearman_delta_feature_vs_delta_score_descriptive": pair_spearman,
        })
    return pd.DataFrame(rows)


def make_patient_aware_summary(
    ps_anchor: pd.DataFrame,
    wp_dir: pd.DataFrame,
) -> pd.DataFrame:
    keep_anchor = [
        "parameter",
        "n_patient_score_rows",
        "n_patients",
        "expected_direction_vs_higher_score",
        "spearman_r_patient_score",
        "spearman_ci95_low_clustered",
        "spearman_ci95_high_clustered",
    ]
    keep_wp = [
        "parameter",
        "n_pairs",
        "n_patients_with_pairs",
        "percent_pairs_expected_direction",
        "percent_pairs_expected_direction_ci95_low_clustered",
        "percent_pairs_expected_direction_ci95_high_clustered",
        "severity_aligned_effect_size_srm",
        "severity_aligned_effect_size_srm_ci95_low_clustered",
        "severity_aligned_effect_size_srm_ci95_high_clustered",
    ]
    return ps_anchor[keep_anchor].merge(wp_dir[keep_wp], on="parameter", how="left")


def format_markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    if df.empty:
        return "<empty>"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}g}")

    def markdown_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(markdown_cell(col) for col in display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(markdown_cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    opts = parse_args()
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    args = get_config(opts.config)

    data_cfg = cfg_get(args, "data")
    analysis_cfg = cfg_get(args, "analysis")
    eval_cfg = cfg_get(args, "eval")

    gait_features = list(cfg_get(data_cfg, "gait_features"))
    expected = get_expected_directions(args, gait_features)
    pair_strategy = str(cfg_get(analysis_cfg, "pair_strategy", "adjacent"))
    n_boot = int(cfg_get(analysis_cfg, "n_boot", 2000))
    seed = int(cfg_get(analysis_cfg, "bootstrap_seed", 123))
    normalize_delta_by_score = bool(cfg_get(analysis_cfg, "normalize_delta_by_score", True))

    raw_path = opts.out_dir / "raw_predictions.csv"

    if opts.predictions is not None:
        print(f"Loading predictions from {opts.predictions}")
        pred_df = pd.read_csv(opts.predictions)
        pred_df.to_csv(raw_path, index=False)
    elif bool(cfg_get(eval_cfg, "reuse_predictions", False)) and raw_path.is_file():
        print(f"Loading existing predictions from {raw_path}")
        pred_df = pd.read_csv(raw_path)
    else:
        pred_df = run_inference(args, opts)
        pred_df.to_csv(raw_path, index=False)
        print(f"Saved raw predictions: {raw_path}")

    if pred_df.empty:
        raise RuntimeError("No predictions available.")

    required = {"patient_id", "updrs_score", *gait_features}
    missing = sorted(required - set(pred_df.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")

    # Ensure consistent types.
    pred_df["patient_id"] = pred_df["patient_id"].astype(str).str.zfill(3)
    pred_df["updrs_score"] = pd.to_numeric(pred_df["updrs_score"], errors="coerce")
    for feat in gait_features:
        pred_df[feat] = pd.to_numeric(pred_df[feat], errors="coerce")

    video_desc = video_level_anchor_validity_descriptive(pred_df, gait_features, expected)
    ps_df = aggregate_patient_score(pred_df, gait_features)
    ps_anchor = patient_score_anchor_validity_clustered(ps_df, gait_features, expected, n_boot=n_boot, seed=seed)
    pairs_df = make_within_patient_pairs(ps_df, gait_features, pair_strategy=pair_strategy)
    wp_dir = within_patient_directional_validity_clustered(
        pairs_df,
        gait_features,
        expected,
        n_boot=n_boot,
        seed=seed,
        normalize_delta_by_score=normalize_delta_by_score,
    )
    summary = make_patient_aware_summary(ps_anchor, wp_dir)

    outputs = {
        "video_level_anchor_validity_descriptive.csv": video_desc,
        "patient_score_predictions.csv": ps_df,
        "patient_score_anchor_validity_clustered.csv": ps_anchor,
        "within_patient_pairs.csv": pairs_df,
        "within_patient_directional_validity_clustered.csv": wp_dir,
        "patient_aware_summary.csv": summary,
    }

    for filename, out_df in outputs.items():
        path = opts.out_dir / filename
        out_df.to_csv(path, index=False)
        print(f"Saved {filename}: {path}")

    print("\n# Patient-aware summary")
    print(format_markdown_table(summary, float_digits=4))

    print("\n# Patient-score aggregated clinical-anchor validity")
    print(format_markdown_table(ps_anchor, float_digits=4))

    print("\n# Within-patient directional validity")
    print(format_markdown_table(wp_dir, float_digits=4))

    print("\nNotes:")
    print("- The video-level table is descriptive only; it does not treat videos as independent for inference.")
    print("- The main correlation table first averages videos within each patient-score level.")
    print("- Confidence intervals use patient-level cluster bootstrap resampling.")
    print("- The within-patient table compares lower-score vs higher-score states of the same patient.")
    print("- These analyses evaluate clinical-anchor/severity validity, not absolute MoCap accuracy.")


if __name__ == "__main__":
    main()


"""
python3 pd4t_patient_aware_clinical_eval.py --config configs/eval/pd4t_patient_aware_clinical_eval.yaml --checkpoint "$GAITXFORMER_CHECKPOINT_FILE" --out-dir pd4t_patient_aware_outputs
"""
