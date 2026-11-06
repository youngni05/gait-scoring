"""Sixteen transparent projected-angle summaries; three train records only."""
import argparse
import ast
import json
from pathlib import Path
import time

import harness as h
from temporal_readiness import ROOT, DATA, MB, sha

FEATURE_NAMES = [side + "_" + name for side in ("right", "left") for name in
                 ("mean_deg", "std_deg", "p10_deg", "p90_deg", "iqr_deg", "mean_abs_frame_change_deg")]
FEATURE_NAMES += ["right_minus_left_mean_deg", "right_minus_left_mean_abs_deg", "right_minus_left_p90_abs_deg", "right_minus_left_std_deg"]


def summaries(motion):
    import numpy as np
    h.require(motion.ndim == 3 and motion.shape[1:] == (17, 3) and len(motion) >= 2 and np.isfinite(motion).all(), "Invalid angle input")
    angles, features = [], []
    for hip, knee, ankle in ((1, 2, 3), (4, 5, 6)):
        a = motion[:, hip, :2].astype(np.float64) - motion[:, knee, :2]
        b = motion[:, ankle, :2].astype(np.float64) - motion[:, knee, :2]
        lengths = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        h.require((lengths > 0).all(), "Zero-length limb segment; do not fabricate an angle")
        cosine = np.sum(a * b, axis=1) / lengths
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        angles.append(angle)
        p10, p25, p75, p90 = np.percentile(angle, [10, 25, 75, 90])
        features.extend([angle.mean(), angle.std(), p10, p90, p75 - p25, np.abs(np.diff(angle)).mean()])
    delta = angles[0] - angles[1]
    features.extend([delta.mean(), np.abs(delta).mean(), np.percentile(np.abs(delta), 90), delta.std()])
    return np.asarray(features), np.stack(angles, axis=1)


def synthetic_checks():
    import numpy as np
    motion = np.zeros((4, 17, 3), dtype=np.float64)
    motion[:, :, 2] = 1
    for joints, degrees in (((1, 2, 3), [90, 80, 70, 60]), ((4, 5, 6), [180, 150, 120, 90])):
        hip, knee, ankle = joints
        theta = np.radians(degrees)
        motion[:, hip, :2] = [0, 1]
        motion[:, ankle, 0] = np.sin(theta)
        motion[:, ankle, 1] = np.cos(theta)
    features, angles = summaries(motion)
    np.testing.assert_allclose(angles, np.array([[90, 180], [80, 150], [70, 120], [60, 90]]), rtol=0, atol=1e-10)
    transformed = motion.copy()
    transformed[:, :, :2] = 3.5 * transformed[:, :, :2] + [14, -9]
    np.testing.assert_allclose(summaries(transformed)[0], features, rtol=0, atol=1e-10)
    swapped = motion.copy()
    swapped[:, [1, 2, 3, 4, 5, 6]] = motion[:, [4, 5, 6, 1, 2, 3]]
    expected = np.r_[features[6:12], features[:6], -features[12], features[13:]]
    np.testing.assert_allclose(summaries(swapped)[0], expected, rtol=0, atol=1e-10)
    bad = motion.copy()
    bad[:, 1] = bad[:, 2]
    try:
        summaries(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero-length segment was accepted")
    return {"known_projected_angles": True, "translation_uniform_scale_invariance": True,
            "left_right_swap_behavior": True, "zero_length_segment_rejected": True}


def run(session):
    started = time.perf_counter()
    import numpy as np
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents, "Invalid session")
    h.require("4단계 계획" in (folder / "plan.md").read_text(encoding="utf-8"), "Missing prior plan")
    h.require(not (folder / "kinematic_readiness_result.json").exists(), "Readiness already completed")
    dev = ROOT / "jobs/development-baseline/runs/20261025T044558572569Z-71adb0"
    packet = dev / "development_packet.npz"
    receipt = h.load(dev / "receipt.json")
    h.require(receipt["status"] == "completed" and sha(dev / "result.json") == receipt["result_sha256"], "Invalid development receipt")
    h.require(sha(packet) == h.load(dev / "result.json")["artifacts"][0]["sha256"], "Packet changed")
    with np.load(packet, allow_pickle=False) as table:
        # Do not access the labels or validation feature arrays.
        videos, people, split = (table[k].copy() for k in ("video_ids", "patient_ids", "split"))
    chosen = sorted(np.flatnonzero(split == "train"), key=lambda i: str(videos[i]))[:3]
    converter = MB / "assets/mmpose_inference_3d.py"
    tracked = [Path(__file__), converter, packet, dev / "result.json", dev / "receipt.json"]
    windows = [DATA / "windows" / (str(videos[i]) + ".npz") for i in chosen]
    tracked += windows
    before = {str(p): sha(p) for p in tracked}
    tree = ast.parse(converter.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "convert_keypoint_definition")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(converter), "exec"), namespace)
    marker = np.arange(51, dtype=np.float32).reshape(1, 17, 3)
    converted = namespace["convert_keypoint_definition"](marker, "coco", "h36m")
    np.testing.assert_array_equal(converted[:, [1, 2, 3]], marker[:, [12, 14, 16]])
    np.testing.assert_array_equal(converted[:, [4, 5, 6]], marker[:, [11, 13, 15]])
    checks = synthetic_checks()
    values, angle_ranges = [], []
    for path in windows:
        with np.load(path, allow_pickle=False) as window:
            motion = window["input"].copy()
        h.require(motion.shape == (243, 17, 3) and (motion[:, 1:7, 2] > 0).all(), "Input observation mismatch")
        feature, angle = summaries(motion)
        h.require(feature.shape == (16,) and np.isfinite(feature).all() and ((angle >= 0) & (angle <= 180)).all(), "Invalid projected-angle output")
        values.append(feature)
        angle_ranges.append([float(angle.min()), float(angle.max())])
    h.require(all(sha(Path(p)) == value for p, value in before.items()), "Source changed")
    output = folder / "three_training_kinematic_probes.npz"
    np.savez_compressed(output, features=np.stack(values), feature_names=np.array(FEATURE_NAMES))
    with np.load(output, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["features"], np.stack(values))
        np.testing.assert_array_equal(saved["feature_names"], np.array(FEATURE_NAMES))
    h.save(folder / "kinematic_sources.json", {"sha256": before, "selected_training_video_ids": [str(videos[i]) for i in chosen],
                "selection": "first three sorted training video IDs; no score selection"})
    result = {"status": "passed", "training_records_checked": 3, "training_people_checked": len(set(people[chosen])),
              "feature_dimensions": 16, "labels_loaded": False, "validation_records_used": 0, "old_test_records_used": 0,
              "score_model_trained": False, "source_files_unchanged": True, "saved_output_reload_passed": True,
              "synthetic_checks": checks, "joint_mapping_synthetic_check": True,
              "angle_interpretation": "projected angles in stored normalized 2D coordinates only; original normalization provenance not re-established",
              "feature_names": FEATURE_NAMES, "elapsed_seconds": time.perf_counter() - started}
    h.save(folder / "kinematic_readiness_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.session), ensure_ascii=True, indent=2))
    except Exception as error:
        import traceback
        h.save(ROOT / "control/desktop-sessions" / args.session / "kinematic_failure.json", {"error_type": type(error).__name__, "details": traceback.format_exc()})
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "details": "saved locally; no automatic retry"}))
        raise SystemExit(1)
