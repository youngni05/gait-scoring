"""Inspect/reproduce representations on three training records, without score fitting."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import harness as h

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "outputs/gdi-ridge-experiment-20261016"
MB = ROOT.parent / "outputs/motionbert-smoke-20261015"
EXPECTED_WEIGHT_SHA256 = "6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run(session):
    started = time.perf_counter()
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents, "작업 폴더 경로 오류")
    h.require((folder / "plan.md").is_file(), "시작 전 계획이 없다.")
    for path in (MB / "vendor", MB / "deps", DATA / "deps"):
        sys.path.insert(0, str(path))
    import numpy as np
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dev = ROOT / "jobs/development-baseline"
    state = h.load(dev / "state.json")
    source = dev / "runs" / state["run_id"]
    receipt = h.load(source / "receipt.json")
    h.require(state["status"] == receipt["status"] == "completed", "개발 자료 생성이 완료되지 않았다.")
    h.require(sha(source / "result.json") == receipt["result_sha256"], "개발 자료 완료 증거 불일치")
    previous = h.load(source / "result.json")
    artifact = previous["artifacts"][0]
    packet = (dev / artifact["path"]).resolve()
    h.require(dev.resolve() in packet.parents and sha(packet) == artifact["sha256"], "개발 묶음 지문/경로 불일치")
    with np.load(packet, allow_pickle=False) as table:
        # Do not load score labels. Select by IDs and split only.
        videos, people, split = table["video_ids"].copy(), table["patient_ids"].copy(), table["split"].copy()
        mean_features = table["X"].copy()
    h.require(mean_features.shape == (243, 512) and set(split) == {"train", "validation"}, "개발 입력 형태 오류")
    train_indices = np.flatnonzero(split == "train")
    h.require(len(train_indices) == 203 and len(set(people[train_indices])) == 175, "학습 분할 수 불일치")
    chosen = sorted(train_indices, key=lambda index: str(videos[index]))[:3]
    weights = MB / "assets/MB_lite.bin"
    converter = MB / "assets/mmpose_inference_3d.py"
    model_source = MB / "vendor/lib/model/DSTformer.py"
    tracked = [packet, source / "result.json", source / "receipt.json", weights, converter, model_source]
    for index in chosen:
        tracked += [DATA / "windows" / (str(videos[index]) + ".npz"), DATA / "features" / (str(videos[index]) + ".npy")]
    before = {str(path): sha(path) for path in tracked}
    h.require(before[str(weights)] == EXPECTED_WEIGHT_SHA256, "기존 모델 가중치 지문과 다름")
    # Use the same converter as original preprocessing, with synthetic joint IDs.
    tree = ast.parse(converter.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "convert_keypoint_definition")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(converter), "exec"), namespace)
    markers = np.arange(17 * 3, dtype=np.float32).reshape(1, 17, 3)
    converted = namespace["convert_keypoint_definition"](markers, "coco", "h36m")
    np.testing.assert_array_equal(converted[:, [1, 2, 3]], markers[:, [12, 14, 16]])
    np.testing.assert_array_equal(converted[:, [4, 5, 6]], markers[:, [11, 13, 15]])
    model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                      num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                      qkv_bias=True, att_fuse=True)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in checkpoint["model_pos"].items()}, strict=True)
    model.requires_grad_(False)
    model.eval()
    h.require(not any(parameter.requires_grad for parameter in model.parameters()), "특징 추출 모델이 고정되지 않음")
    rows, pooled, temporal, sides = [], [], [], []
    for number, index in enumerate(chosen, 1):
        h.require(split[index] == "train", "검사 표본이 학습 자료가 아님")
        window_file = DATA / "windows" / (str(videos[index]) + ".npz")
        feature_file = DATA / "features" / (str(videos[index]) + ".npy")
        with np.load(window_file, allow_pickle=False) as window:
            motion = window["input"].copy()
        saved = np.load(feature_file, allow_pickle=False)
        h.require(motion.shape == (243, 17, 3) and motion.dtype == np.float32 and np.isfinite(motion).all(), "입력 형태/수치 오류")
        h.require((motion[:, 1:7, 2] > 0).all(), "하체 관측 조건 불일치")
        np.testing.assert_array_equal(saved, mean_features[index])
        with torch.inference_mode():
            representation = model.get_representation(torch.from_numpy(motion[None]))
            h.require(tuple(representation.shape) == (1, 243, 17, 512) and torch.isfinite(representation).all().item(), "모델 출력 형태/수치 오류")
            base = representation.mean(dim=(1, 2)).numpy()[0]
            by_time = torch.stack([representation[:, offset:offset + 81].mean(dim=(1, 2))[0] for offset in (0, 81, 162)]).numpy()
            by_side = torch.stack([representation[:, :, joints].mean(dim=(1, 2))[0] for joints in ([1, 2, 3], [4, 5, 6])]).numpy()
        np.testing.assert_allclose(base, saved, rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(by_time.mean(axis=0), base, rtol=1e-5, atol=2e-6)
        h.require(by_time.shape == (3, 512) and by_side.shape == (2, 512) and np.isfinite(by_time).all() and np.isfinite(by_side).all(), "나눈 표현 형태/수치 오류")
        pooled.append(base)
        temporal.append(by_time)
        sides.append(by_side)
        rows.append({"sample_number": number, "split": "train", "window_source": str(window_file),
                     "mean_feature_max_abs_difference": float(np.max(np.abs(base - saved))),
                     "three_time_means_identical": bool(np.array_equal(by_time[0], by_time[1]) and np.array_equal(by_time[1], by_time[2])),
                     "right_left_means_identical": bool(np.array_equal(by_side[0], by_side[1]))})
    h.require(all(sha(Path(path)) == value for path, value in before.items()), "검사 도중 원본 지문 변경")
    np.savez_compressed(folder / "three_training_feature_probes.npz", pooled=np.stack(pooled),
                        time_thirds=np.stack(temporal), right_left_legs=np.stack(sides))
    result = {"status": "passed", "records_checked": len(chosen), "training_people_checked": len(set(people[chosen])),
              "selection": "First three video IDs in existing train split; no score-based selection",
              "labels_loaded": False, "validation_records_used": 0, "previous_test_records_used": 0,
              "input_shape": [243, 17, 3], "representation_shape": [1, 243, 17, 512],
              "pooled_feature_dimensions": 512, "time_thirds_dimensions": 1536, "right_left_leg_dimensions": 1024,
              "weight_sha256_matched": True, "model_frozen": True, "side_mapping_checked_with_synthetic_markers": True,
              "all_mean_features_reproduced": True, "max_mean_feature_abs_difference": max(row["mean_feature_max_abs_difference"] for row in rows),
              "source_files_unchanged": True, "score_model_trained": False, "accuracy_improvement_evaluated": False,
              "elapsed_seconds": time.perf_counter() - started, "versions": {"numpy": np.__version__, "torch": torch.__version__}}
    h.save(folder / "source_manifest.json", {"sha256": before, "selected_training_samples": rows})
    h.save(folder / "readiness_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.session), ensure_ascii=False, indent=2))
