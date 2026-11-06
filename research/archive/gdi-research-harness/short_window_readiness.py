"""Native 124-frame inference on three existing train records; no score labels."""
import argparse
import ast
import json
from pathlib import Path
import sys
import time

import harness as h
from temporal_readiness import ROOT, DATA, MB, sha, EXPECTED_WEIGHT_SHA256


def validate_short(motion):
    import numpy as np
    h.require(motion.shape == (124, 17, 3) and motion.dtype == np.float32, "Expected a 124-frame float32 probe")
    h.require(np.isfinite(motion).all() and (motion[:, 1:7, 2] > 0).all(), "Invalid or unobserved lower-body input")
    h.require(((motion[..., 2] >= 0) & (motion[..., 2] <= 1)).all(), "Invalid confidence values")


def run(session):
    started = time.perf_counter()
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents, "Invalid session path")
    h.require((folder / "plan.md").is_file(), "Missing prior plan")
    h.require(not (folder / "short_readiness_result.json").exists(), "Readiness already saved; do not repeat")
    for path in (MB / "vendor", MB / "deps", DATA / "deps"):
        sys.path.insert(0, str(path))
    import numpy as np
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dev = ROOT / "jobs/development-baseline/runs/20261025T044558572569Z-71adb0"
    receipt = h.load(dev / "receipt.json")
    h.require(receipt["status"] == "completed" and sha(dev / "result.json") == receipt["result_sha256"], "Invalid development receipt")
    packet = dev / "development_packet.npz"
    h.require(sha(packet) == h.load(dev / "result.json")["artifacts"][0]["sha256"], "Packet changed")
    with np.load(packet, allow_pickle=False) as table:
        # NPZ lazy access: never access Y or the validation feature matrix.
        videos, people, split = (table[k].copy() for k in ("video_ids", "patient_ids", "split"))
    train = np.flatnonzero(split == "train")
    h.require(len(train) == 203 and len(set(people[train])) == 175, "Wrong training population")
    chosen = sorted(train, key=lambda i: str(videos[i]))[:3]
    weights = MB / "assets/MB_lite.bin"
    preparation = ROOT.parent / "tools/stanford-sample-check/prepare_ridge_data.py"
    model_source = MB / "vendor/lib/model/DSTformer.py"
    windows = [DATA / "windows" / (str(videos[i]) + ".npz") for i in chosen]
    old_features = [DATA / "features" / (str(videos[i]) + ".npy") for i in chosen]
    tracked = [packet, dev / "receipt.json", dev / "result.json", weights, preparation, Path(__file__), ROOT / "temporal_readiness.py"]
    tracked += sorted((MB / "vendor/lib/model").rglob("*.py")) + windows + old_features
    before = {str(p): sha(p) for p in tracked}
    h.require(before[str(weights)] == EXPECTED_WEIGHT_SHA256, "Frozen weight mismatch")
    tree = ast.parse(preparation.read_text(encoding="utf-8"))
    normalizer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "normalize")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[normalizer], type_ignores=[]), str(preparation), "exec"), namespace)
    normalize = namespace["normalize"]
    h.save(folder / "normalization_function.py", "import numpy as np\n\n" + ast.unparse(normalizer) + "\n")
    text = model_source.read_text(encoding="utf-8")
    h.require("self.temp_embed[:,:F,:,:]" in text, "Expected dynamic time-position slicing missing")
    h.save(folder / "source_inspection.json", {"model_source": str(model_source), "sha256": sha(model_source),
                 "dynamic_length_expression": "self.temp_embed[:,:F,:,:]", "weight_maxlen": 243,
                 "claim": "implementation slices temporal positions to actual F; inference support still requires execution"})
    model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                      num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243, qkv_bias=True, att_fuse=True)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["model_pos"].items()}, strict=True)
    model.requires_grad_(False)
    model.eval()
    h.require(not any(p.requires_grad for p in model.parameters()), "Model not frozen")
    crop_inputs, normalized_inputs, crop_features, normalized_features, references = [], [], [], [], []
    rows = []
    for number, (window_file, old_file) in enumerate(zip(windows, old_features), 1):
        with np.load(window_file, allow_pickle=False) as window:
            motion, frame_ids = window["input"].copy(), window["frame_ids"].copy()
        h.require(motion.shape == (243, 17, 3) and motion.dtype == np.float32 and np.isfinite(motion).all(), "Invalid original window")
        h.require(frame_ids.shape == (243,) and (np.diff(frame_ids) == 1).all(), "Noncontiguous source frame indices")
        np.testing.assert_allclose(normalize(motion), motion, rtol=1e-5, atol=2e-6)
        crop = motion[:124].copy()
        short_normalized = normalize(crop)
        for short in (crop, short_normalized):
            validate_short(short)
        np.testing.assert_array_equal(short_normalized[..., 2], crop[..., 2])
        h.require(np.abs(short_normalized[..., :2]).max() <= 1, "Short normalized coordinates outside range")
        outputs = []
        timings = []
        for short in (crop, short_normalized):
            inference_start = time.perf_counter()
            with torch.inference_mode():
                rep = model.get_representation(torch.from_numpy(short[None]))
                h.require(tuple(rep.shape) == (1, 124, 17, 512) and torch.isfinite(rep).all().item(), "124-frame output invalid")
                pooled = rep.mean(dim=(1, 2)).numpy()[0].copy()
            h.require(pooled.shape == (512,) and np.isfinite(pooled).all(), "Invalid pooled feature")
            outputs.append(pooled)
            timings.append(time.perf_counter() - inference_start)
        full = np.load(old_file, allow_pickle=False)
        h.require(full.shape == (512,) and np.isfinite(full).all(), "Invalid old reference")
        crop_inputs.append(crop)
        normalized_inputs.append(short_normalized)
        crop_features.append(outputs[0])
        normalized_features.append(outputs[1])
        references.append(full)
        rows.append({"sample_number": number, "video_id": str(videos[chosen[number - 1]]), "split": "train",
                     "crop_vs_short_normalized_max_coordinate_difference": float(np.max(np.abs(crop[..., :2] - short_normalized[..., :2]))),
                     "crop_vs_short_normalized_max_feature_difference": float(np.max(np.abs(outputs[0] - outputs[1]))),
                     "full_vs_short_normalized_feature_mean_abs_difference": float(np.mean(np.abs(full - outputs[1]))),
                     "inference_seconds_crop_and_renormalized": timings})
    h.require(all(sha(Path(p)) == value for p, value in before.items()), "Source changed during inference")
    arrays = {"crop_inputs": np.stack(crop_inputs), "normalized_inputs": np.stack(normalized_inputs),
              "features_crop": np.stack(crop_features), "features_normalized": np.stack(normalized_features),
              "reference_full243_features": np.stack(references)}
    np.savez_compressed(folder / "short_training_probes.npz", **arrays)
    with np.load(folder / "short_training_probes.npz", allow_pickle=False) as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    h.save(folder / "short_sources.json", {"sha256": before, "selected_training_records": rows,
                 "selection": "first three sorted original training video IDs, no score selection"})
    result = {"status": "passed", "training_records_checked": 3, "training_people_checked": len(set(people[chosen])),
              "input_frames": 124, "weight_maxlen_unchanged": 243, "representation_shape": [1, 124, 17, 512],
              "pooled_dimensions": 512, "normalization_conditions": ["crop_only_existing243_scale", "normalize_cropped124_again"],
              "original_full_window_normalization_reproduced": True, "confidence_values_preserved": True,
              "contiguous_frame_ids_verified": True, "source_files_unchanged": True,
              "labels_loaded": False, "validation_records_used": 0, "old_test_records_used": 0,
              "new_people_added": 0, "score_model_trained": False, "accuracy_improvement_evaluated": False,
              "raw_short_clip_equivalence_proved": False, "elapsed_seconds": time.perf_counter() - started}
    h.save(folder / "short_readiness_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.session), ensure_ascii=True, indent=2))
    except Exception as error:
        import traceback
        h.save(ROOT / "control/desktop-sessions" / args.session / "short_readiness_failure.json", {"error_type": type(error).__name__, "details": traceback.format_exc()})
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "details": "saved locally; no automatic retry"}))
        raise SystemExit(1)
