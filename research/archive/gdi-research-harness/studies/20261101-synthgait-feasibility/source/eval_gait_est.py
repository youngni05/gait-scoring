"""Evaluate GaitXFormer on cached SynthGait validation data or raw GPJATK."""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cached_gait_est import CachedGaitEstimationDataset
from data.gpjatk import (
    GPJATK_ALL_VIEW_NAMES,
    GPJATKDataset,
    gpjatk_view_name,
    normalize_eval_view,
    preferred_gpjatk_views,
)
from model.gaitxformer import GaitXFormer
from utils.tools import get_config, print_args, resolve_checkpoint_path


RANGES_OF_VALUES = {
    "cadence": [32.9268, 211.7647],
    "walking_speed": [0.0008, 1.4817],
    "step_length_mean": [0.0771, 0.8621],
    "step_width_mean": [0.0012, 0.2803],
    "stoop_posture_mean": [-0.1116, 0.3497],
    "arm_swing": [0.0202, 0.6795],
}
VALIDATION_VIEW_BY_TOKEN = {
    "_frontal": "front",
    "_saggital": "side",
    "_random": "oblique",
    "_view1": "front",
    "_view2": "side",
    "_view3": "oblique",
}
VALIDATION_ALL_VIEW_NAMES = ("front", "side", "oblique")


def fisher_avg(values):
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    epsilon = np.finfo(float).eps
    clipped_values = [
        max(min(value, 1.0 - epsilon), -1.0 + epsilon)
        for value in finite_values
    ]
    return math.tanh(
        sum(math.atanh(value) for value in clipped_values) / len(clipped_values)
    )


def normalized_mae(predictions, targets, feature_name):
    mae = np.mean(np.abs(predictions - targets))
    if feature_name in RANGES_OF_VALUES:
        feature_min, feature_max = RANGES_OF_VALUES[feature_name]
        feature_range = feature_max - feature_min
    else:
        feature_range = np.max(targets) - np.min(targets)
    return mae / feature_range if feature_range else float("inf")


def pearsonr_safe(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.size < 2:
        return float("nan")
    if np.std(predictions, ddof=1) == 0 or np.std(targets, ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(predictions, targets)[0, 1])


def visualize_corr(predictions, targets, feature_name, correlation, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, predictions, alpha=0.6, s=20)
    plt.xlabel(f"Ground Truth {feature_name}")
    plt.ylabel(f"Predicted {feature_name}")
    plt.title(f"{feature_name} Correlation: {correlation:.3f}")
    min_value = min(targets.min(), predictions.min())
    max_value = max(targets.max(), predictions.max())
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "r--",
        alpha=0.8,
        linewidth=2,
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def dataset_video_name(dataset, dataset_index):
    return dataset.sample_name(dataset_index)


def validation_view_name(video_file):
    video_stem = os.path.splitext(os.path.basename(str(video_file)))[0]
    for view_token, view_name in VALIDATION_VIEW_BY_TOKEN.items():
        if video_stem.endswith(view_token):
            return view_name
    return None


def video_view_name(eval_data, video_file):
    if eval_data == "validation":
        return validation_view_name(video_file)
    return gpjatk_view_name(video_file)


def dataset_view_names(dataset, dataset_indices, eval_data):
    return np.array(
        [
            video_view_name(
                eval_data,
                dataset_video_name(dataset, dataset_index),
            )
            for dataset_index in dataset_indices
        ],
        dtype=object,
    )


def preferred_validation_views(feature_name):
    normalized_name = str(feature_name).lower()
    if normalized_name == "cadence":
        return VALIDATION_ALL_VIEW_NAMES
    if "step_width" in normalized_name or "width" in normalized_name:
        return ("front",)
    return ("side",)


def all_metric_view_groups(eval_data):
    if eval_data == "validation":
        return VALIDATION_ALL_VIEW_NAMES
    return GPJATK_ALL_VIEW_NAMES


def selected_metric_view_groups(eval_view, feature_name, eval_data):
    if eval_view == "all":
        return None
    if eval_view == "avg":
        return all_metric_view_groups(eval_data)
    if eval_view == "preferred":
        if eval_data == "validation":
            return preferred_validation_views(feature_name)
        return preferred_gpjatk_views(feature_name)
    return (eval_view,)


def mask_for_view_groups(view_names, view_groups):
    if view_groups is None:
        return np.ones(len(view_names), dtype=bool)
    return np.isin(view_names, list(view_groups))


def mean_by_view(values, view_names, view_groups, reducer):
    if view_groups is None:
        return float(reducer(values))
    reduced_values = [
        float(reducer(values[view_names == view_group]))
        for view_group in view_groups
        if np.any(view_names == view_group)
    ]
    return float(np.mean(reduced_values)) if reduced_values else float("nan")


def nmae_for_view_groups(
    predictions,
    targets,
    feature_name,
    view_names,
    view_groups,
):
    if view_groups is None:
        return normalized_mae(predictions, targets, feature_name)
    values = [
        normalized_mae(
            predictions[view_names == view_group],
            targets[view_names == view_group],
            feature_name,
        )
        for view_group in view_groups
        if np.any(view_names == view_group)
    ]
    return float(np.mean(values)) if values else float("nan")


def corr_for_view_groups(predictions, targets, view_names, view_groups):
    if view_groups is None:
        return pearsonr_safe(predictions, targets)
    return fisher_avg(
        [
            pearsonr_safe(
                predictions[view_names == view_group],
                targets[view_names == view_group],
            )
            for view_group in view_groups
            if np.any(view_names == view_group)
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/eval/hmgpjatk.yaml",
        help="Evaluation YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        required=True,
        help="Checkpoint file or directory containing best_ckpt.pth.tr",
    )
    parser.add_argument(
        "--view",
        choices=("all", "side", "front", "back", "oblique", "preferred", "avg"),
        default=None,
        help="Override data.view without editing the YAML config",
    )
    parser.add_argument(
        "--corr-dir",
        default="correlation_plots",
        help="Directory for per-video outputs, summaries, and plots",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(args, model, data_loader, device, corr_dir):
    model.eval()
    os.makedirs(corr_dir, exist_ok=True)
    all_outputs = []
    all_targets = []
    all_indices = []
    total_videos = 0
    evaluated_videos = 0
    skipped_short_videos = 0
    running_dataset_index = 0

    for videos, valids, targets in tqdm(data_loader):
        videos = videos.to(device, non_blocking=True)
        valids = valids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        valid_counts = valids.sum(dim=1)
        keep_mask = valid_counts >= args.min_valid_frames
        batch_total = valid_counts.numel()
        batch_indices = torch.arange(
            running_dataset_index,
            running_dataset_index + batch_total,
            dtype=torch.long,
        )
        running_dataset_index += batch_total
        batch_evaluated = int(keep_mask.sum().item())
        total_videos += batch_total
        evaluated_videos += batch_evaluated
        skipped_short_videos += batch_total - batch_evaluated

        videos = videos[keep_mask]
        valids = valids[keep_mask]
        targets = targets[keep_mask]
        batch_indices = batch_indices[keep_mask.cpu()]
        if not videos.size(0):
            continue

        outputs = model(videos, valids)
        if outputs.shape != targets.shape:
            raise RuntimeError(
                f"Model output shape {tuple(outputs.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )
        all_outputs.append(
            data_loader.dataset.denormalize_gait_features(outputs).cpu()
        )
        all_targets.append(
            data_loader.dataset.denormalize_gait_features(targets).cpu()
        )
        all_indices.append(batch_indices)

    print(
        f"{evaluated_videos}/{total_videos} videos evaluated, "
        f"{skipped_short_videos} skipped due to being short"
    )
    if not all_outputs:
        raise RuntimeError("No videos passed the minimum-valid-frame threshold.")

    outputs = torch.cat(all_outputs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    dataset_indices = torch.cat(all_indices, dim=0).numpy()
    eval_view = normalize_eval_view(args.data.view)
    view_names = dataset_view_names(
        data_loader.dataset,
        dataset_indices,
        args.eval_data,
    )
    if eval_view != "all" and not any(name is not None for name in view_names):
        raise ValueError(
            f"view='{eval_view}' requires parseable view tokens in video filenames."
        )

    feature_metrics = {}
    for feature_index, feature_name in enumerate(args.data.gait_features):
        view_groups = selected_metric_view_groups(
            eval_view,
            feature_name,
            args.eval_data,
        )
        selected_mask = mask_for_view_groups(view_names, view_groups)
        feature_predictions = outputs[:, feature_index]
        feature_targets = targets[:, feature_index]
        correlation = corr_for_view_groups(
            feature_predictions,
            feature_targets,
            view_names,
            view_groups,
        )
        mae = mean_by_view(
            np.abs(feature_predictions - feature_targets),
            view_names,
            view_groups,
            np.mean,
        )
        nmae = nmae_for_view_groups(
            feature_predictions,
            feature_targets,
            feature_name,
            view_names,
            view_groups,
        )
        bias = mean_by_view(
            feature_predictions - feature_targets,
            view_names,
            view_groups,
            np.mean,
        )
        feature_metrics[feature_name] = {
            "correlation": correlation,
            "mae": mae,
            "nmae": nmae,
            "bias": bias,
            "view_groups": view_groups,
            "selected_mask": selected_mask,
        }

        selected_predictions = feature_predictions[selected_mask]
        selected_targets = feature_targets[selected_mask]
        if selected_predictions.size:
            visualize_corr(
                selected_predictions,
                selected_targets,
                feature_name,
                correlation,
                os.path.join(corr_dir, f"correlation_{feature_name}.png"),
            )

        prediction_rows = {}
        for dataset_index, prediction, target, view_name, selected in zip(
            dataset_indices,
            feature_predictions,
            feature_targets,
            view_names,
            selected_mask,
        ):
            if not selected:
                continue
            row = {
                "pred": float(prediction),
                "gt": float(target),
                "video_name": dataset_video_name(
                    data_loader.dataset,
                    dataset_index,
                ),
            }
            if view_name is not None:
                row["view"] = view_name
            prediction_rows[str(int(dataset_index))] = row
        with open(
            os.path.join(corr_dir, f"pred_gt_{feature_name}.json"),
            "w",
        ) as output_file:
            json.dump(prediction_rows, output_file, indent=2)

    overall_correlation = fisher_avg(
        [metrics["correlation"] for metrics in feature_metrics.values()]
    )
    overall_mae = float(
        np.mean([metrics["mae"] for metrics in feature_metrics.values()])
    )
    overall_nmae = float(
        np.mean([metrics["nmae"] for metrics in feature_metrics.values()])
    )

    write_correlation_summary(
        corr_dir,
        eval_view,
        args,
        feature_metrics,
        outputs,
        targets,
        view_names,
        overall_correlation,
    )
    write_error_summary(
        corr_dir,
        eval_view,
        args,
        feature_metrics,
        outputs,
        targets,
        view_names,
    )
    write_bias_summary(
        corr_dir,
        eval_view,
        args,
        feature_metrics,
        outputs,
        targets,
        view_names,
    )

    serializable_metrics = {
        "view_mode": eval_view,
        "correlation_unit": "per-video",
        "evaluated_videos": evaluated_videos,
        "skipped_short_videos": skipped_short_videos,
        "overall": {
            "correlation_fisher_average": overall_correlation,
            "mae_mean": overall_mae,
            "nmae_mean": overall_nmae,
        },
        "features": {
            feature_name: {
                key: value
                for key, value in metrics.items()
                if key not in {"selected_mask", "view_groups"}
            }
            | {
                "view_groups": list(metrics["view_groups"])
                if metrics["view_groups"] is not None
                else None
            }
            for feature_name, metrics in feature_metrics.items()
        },
    }
    with open(os.path.join(corr_dir, "metrics.json"), "w") as output_file:
        json.dump(serializable_metrics, output_file, indent=2, allow_nan=True)

    result = {
        f"eval/corr_{name}": metrics["correlation"]
        for name, metrics in feature_metrics.items()
    }
    result.update(
        {
            f"eval/mae_{name}": metrics["mae"]
            for name, metrics in feature_metrics.items()
        }
    )
    result.update(
        {
            f"eval/nmae_{name}": metrics["nmae"]
            for name, metrics in feature_metrics.items()
        }
    )
    result.update(
        {
            f"eval/bias_{name}": metrics["bias"]
            for name, metrics in feature_metrics.items()
        }
    )
    result.update(
        {
            "eval/corr_all": overall_correlation,
            "eval/mae_all": overall_mae,
            "eval/nmae_all": overall_nmae,
        }
    )
    return result


def view_lines(view_groups, feature_index, outputs, targets, view_names):
    if view_groups is None:
        return []
    lines = []
    for view_group in view_groups:
        view_mask = view_names == view_group
        if not np.any(view_mask):
            continue
        predictions = outputs[view_mask, feature_index]
        ground_truth = targets[view_mask, feature_index]
        lines.append(
            {
                "view": view_group,
                "correlation": pearsonr_safe(predictions, ground_truth),
                "mae": float(np.mean(np.abs(predictions - ground_truth))),
                "n": int(np.sum(view_mask)),
                "bias": float(np.mean(predictions - ground_truth)),
            }
        )
    return lines


def write_correlation_summary(
    corr_dir,
    eval_view,
    args,
    feature_metrics,
    outputs,
    targets,
    view_names,
    overall_correlation,
):
    with open(os.path.join(corr_dir, "correlation_summary.txt"), "w") as output_file:
        output_file.write("Gait Feature Correlation Summary\n")
        output_file.write("=" * 80 + "\n\n")
        output_file.write(f"View mode: {eval_view}\n")
        if eval_view == "avg":
            output_file.write(
                "Averaging: per-view correlations combined with a Fisher z-transform.\n"
            )
        output_file.write("Correlation unit: per-video\n\n")
        for index, feature_name in enumerate(args.data.gait_features):
            metrics = feature_metrics[feature_name]
            output_file.write(f"Feature: {feature_name}\n")
            if metrics["view_groups"] is not None:
                output_file.write(
                    f"  View groups: {', '.join(metrics['view_groups'])}\n"
                )
                output_file.write(
                    f"  Pearson Fisher average: r={metrics['correlation']:.4f}\n"
                )
                output_file.write("  Per-view values:\n")
                for values in view_lines(
                    metrics["view_groups"], index, outputs, targets, view_names
                ):
                    output_file.write(
                        f"    {values['view']}: r={values['correlation']:.4f}, "
                        f"N={values['n']}\n"
                    )
            else:
                output_file.write(
                    f"  Pearson correlation: r={metrics['correlation']:.4f}\n"
                )
            output_file.write("\n")
        output_file.write("Overall correlation:\n")
        output_file.write(
            f"  Pearson Fisher average: r={overall_correlation:.4f}\n"
        )


def write_error_summary(
    corr_dir,
    eval_view,
    args,
    feature_metrics,
    outputs,
    targets,
    view_names,
):
    with open(os.path.join(corr_dir, "nmae_summary.txt"), "w") as output_file:
        output_file.write("Gait Feature Normalized Mean Absolute Error (NMAE) Summary\n")
        output_file.write("=" * 80 + "\n\n")
        output_file.write(f"View mode: {eval_view}\n")
        if eval_view == "avg":
            output_file.write("Averaging: per-view MAE/NMAE arithmetic mean.\n")
        output_file.write("\n")
        for index, feature_name in enumerate(args.data.gait_features):
            metrics = feature_metrics[feature_name]
            selected_mask = metrics["selected_mask"]
            predictions = outputs[selected_mask, index]
            ground_truth = targets[selected_mask, index]
            output_file.write(f"Feature: {feature_name}\n")
            if metrics["view_groups"] is not None:
                output_file.write(
                    f"  View groups: {', '.join(metrics['view_groups'])}\n"
                )
            output_file.write(f"  MAE: {metrics['mae']:.4f}\n")
            output_file.write(f"  NMAE: {metrics['nmae']:.4f}\n")
            if ground_truth.size:
                output_file.write(
                    f"  Ground Truth - Min: {np.min(ground_truth):.4f}, "
                    f"Max: {np.max(ground_truth):.4f}, "
                    f"Range: {np.ptp(ground_truth):.4f}\n"
                )
                output_file.write(
                    f"  Ground Truth - Mean: {np.mean(ground_truth):.4f}, "
                    f"Std: {np.std(ground_truth):.4f}\n"
                )
                output_file.write(
                    f"  Predicted - Mean: {np.mean(predictions):.4f}, "
                    f"Std: {np.std(predictions):.4f}\n"
                )
            output_file.write(f"  N samples: {ground_truth.size}\n")
            if metrics["view_groups"] is not None:
                output_file.write("  Per-view values:\n")
                for values in view_lines(
                    metrics["view_groups"], index, outputs, targets, view_names
                ):
                    view_mask = view_names == values["view"]
                    view_nmae = normalized_mae(
                        outputs[view_mask, index],
                        targets[view_mask, index],
                        feature_name,
                    )
                    output_file.write(
                        f"    {values['view']}: MAE={values['mae']:.4f}, "
                        f"NMAE={view_nmae:.4f}, N={values['n']}\n"
                    )
            output_file.write("\n")


def write_bias_summary(
    corr_dir,
    eval_view,
    args,
    feature_metrics,
    outputs,
    targets,
    view_names,
):
    with open(os.path.join(corr_dir, "bias_summary.txt"), "w") as output_file:
        output_file.write("Gait Feature Bias (Mean Signed Error) Summary\n")
        output_file.write("=" * 80 + "\n\n")
        output_file.write("Definition: bias = mean(prediction - ground truth).\n")
        output_file.write(
            "Interpretation: positive = overprediction; negative = underprediction.\n"
        )
        output_file.write(f"View mode: {eval_view}\n")
        if eval_view == "avg":
            output_file.write("Averaging: per-view bias arithmetic mean.\n")
        output_file.write("\n")
        for index, feature_name in enumerate(args.data.gait_features):
            metrics = feature_metrics[feature_name]
            output_file.write(f"Feature: {feature_name}\n")
            if metrics["view_groups"] is not None:
                output_file.write(
                    f"  View groups: {', '.join(metrics['view_groups'])}\n"
                )
            output_file.write(f"  Bias: {metrics['bias']:.4f}\n")
            output_file.write(
                f"  N samples: {int(np.sum(metrics['selected_mask']))}\n"
            )
            if metrics["view_groups"] is not None:
                output_file.write("  Per-view values:\n")
                for values in view_lines(
                    metrics["view_groups"], index, outputs, targets, view_names
                ):
                    output_file.write(
                        f"    {values['view']}: Bias={values['bias']:.4f}, "
                        f"N={values['n']}\n"
                    )
            output_file.write("\n")


def run_evaluation(args, opts):
    print_args(args)
    if args.eval_data not in {"gpjatk", "validation"}:
        raise ValueError("The retained evaluator supports gpjatk and validation only.")
    data_args = dict(args.data)
    if args.eval_data == "validation":
        data_args.pop("view", None)
        dataset = CachedGaitEstimationDataset(**data_args)
    else:
        dataset = GPJATKDataset(**data_args)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitXFormer(**args.model, n_out=len(args.data.gait_features))
    model.to(device)
    checkpoint_path = resolve_checkpoint_path(opts.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)

    eval_result = evaluate(args, model, data_loader, device, opts.corr_dir)
    for key, value in eval_result.items():
        print(f"{key}: {value}")


def main():
    opts = parse_args()
    args = get_config(opts.config)
    if opts.view is not None:
        args.data.view = opts.view
    run_evaluation(args, opts)


if __name__ == "__main__":
    main()
