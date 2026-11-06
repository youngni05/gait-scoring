"""Bounded, resumable frozen representations; never load score labels."""
import argparse
import os
import sys
import time
from pathlib import Path

import harness as h
from temporal_readiness import ROOT, DATA, MB, EXPECTED_WEIGHT_SHA256, sha


def run(session):
    started = time.perf_counter()
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents, "Invalid session")
    h.require((folder / "plan.md").is_file(), "Missing prior plan")
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
    h.require(state["status"] == receipt["status"] == "completed", "Incomplete development packet")
    h.require(sha(source / "result.json") == receipt["result_sha256"], "Invalid development receipt")
    artifact = h.load(source / "result.json")["artifacts"][0]
    packet = (dev / artifact["path"]).resolve()
    h.require(dev.resolve() in packet.parents and sha(packet) == artifact["sha256"], "Invalid packet hash")
    with np.load(packet, allow_pickle=False) as table:
        # NPZ is lazy: do not access Y or any other original evaluation table.
        videos, people, split, original = (table[k].copy() for k in ("video_ids", "patient_ids", "split", "X"))
    train, val = split == "train", split == "validation"
    h.require(original.shape == (243, 512) and len(set(videos)) == 243, "Wrong record counts")
    h.require(train.sum() == 203 and val.sum() == 40, "Wrong splits")
    h.require(len(set(people[train])) == 175 and len(set(people[val])) == 37, "Wrong person counts")
    h.require(not (set(people[train]) & set(people[val])), "Person overlap")
    windows, old_features = [], []
    for video in videos:
        h.require(Path(str(video)).name == str(video), "Invalid record filename")
        windows.append(DATA / "windows" / (str(video) + ".npz"))
        old_features.append(DATA / "features" / (str(video) + ".npy"))
    weights = MB / "assets/MB_lite.bin"
    tracked = [packet, source / "result.json", source / "receipt.json", weights,
               MB / "assets/mmpose_inference_3d.py", ROOT / "temporal_readiness.py", Path(__file__),
               ROOT / "harness.py"]
    tracked += sorted((MB / "vendor/lib/model").rglob("*.py"))
    tracked += windows + old_features
    before = {str(path): sha(path) for path in tracked}
    h.require(before[str(weights)] == EXPECTED_WEIGHT_SHA256, "Weight hash mismatch")
    config = {"version": 1, "threads": 4, "rtol": 1e-5, "atol": 2e-6,
              "time_offsets": [0, 81, 162], "frames_per_third": 81,
              "right_joints": [1, 2, 3], "left_joints": [4, 5, 6],
              "numpy": np.__version__, "torch": torch.__version__, "source_sha256": before}
    fingerprint = h.digest(__import__("json").dumps(config, sort_keys=True).encode())
    manifest_file = folder / "cache_manifest.json"
    if manifest_file.exists():
        manifest = h.load(manifest_file)
        h.require(manifest["fingerprint"] == fingerprint, "Cache source/config changed; preserve and stop")
    else:
        manifest = {"fingerprint": fingerprint, "config": config, "records": {}}
        h.save(manifest_file, manifest)
    model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                      num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243, qkv_bias=True, att_fuse=True)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["model_pos"].items()}, strict=True)
    model.requires_grad_(False)
    model.eval()
    h.require(not any(p.requires_grad for p in model.parameters()), "Model not frozen")
    shapes = {"global_mean": (512,), "time_thirds": (1536,), "leg_mean": (512,), "right_left": (1024,)}

    def inspect(arrays, index):
        for key, shape in shapes.items():
            h.require(arrays[key].shape == shape and arrays[key].dtype == np.float32
                      and np.isfinite(arrays[key]).all(), "Invalid cached shape/value")
        np.testing.assert_allclose(arrays["global_mean"], original[index], rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(arrays["time_thirds"].reshape(3, 512).mean(0), arrays["global_mean"], rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(arrays["right_left"].reshape(2, 512).mean(0), arrays["leg_mean"], rtol=1e-5, atol=2e-6)

    reused, computed = 0, 0
    cache = folder / "records"
    cache.mkdir(exist_ok=True)
    for index in range(len(videos)):
        key = str(index)
        destination = cache / f"row_{index:04d}.npz"
        if key in manifest["records"]:
            entry = manifest["records"][key]
            h.require(entry["video_id"] == str(videos[index]) and entry["sha256"] == sha(destination), "Cached record mismatch")
            with np.load(destination, allow_pickle=False) as saved:
                inspect(saved, index)
            reused += 1
            continue
        if time.perf_counter() - started >= 420:
            break
        h.require(not destination.exists(), "Uncommitted cache file; inspect before continuing")
        with np.load(windows[index], allow_pickle=False) as window:
            motion = window["input"].copy()
        h.require(motion.shape == (243, 17, 3) and motion.dtype == np.float32 and np.isfinite(motion).all(), "Invalid input")
        h.require((motion[:, 1:7, 2] > 0).all(), "Lower body observation mismatch")
        np.testing.assert_array_equal(np.load(old_features[index], allow_pickle=False), original[index])
        with torch.inference_mode():
            rep = model.get_representation(torch.from_numpy(motion[None]))
            h.require(tuple(rep.shape) == (1, 243, 17, 512) and torch.isfinite(rep).all().item(), "Invalid representation")
            arrays = {
                "global_mean": rep.mean(dim=(1, 2)).numpy()[0],
                "time_thirds": torch.stack([rep[:, t:t+81].mean(dim=(1, 2))[0] for t in (0, 81, 162)]).numpy().reshape(-1),
                "leg_mean": rep[:, :, [1, 2, 3, 4, 5, 6]].mean(dim=(1, 2)).numpy()[0],
                "right_left": torch.stack([rep[:, :, j].mean(dim=(1, 2))[0] for j in ([1, 2, 3], [4, 5, 6])]).numpy().reshape(-1),
            }
        inspect(arrays, index)
        temp = destination.with_suffix(".tmp")
        with temp.open("wb") as out:
            np.savez_compressed(out, **arrays)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, destination)
        with np.load(destination, allow_pickle=False) as saved:
            for name in shapes:
                np.testing.assert_array_equal(saved[name], arrays[name])
        manifest["records"][key] = {"video_id": str(videos[index]), "split": str(split[index]),
                                    "sha256": sha(destination),
                                    "max_mean_difference": float(np.max(np.abs(arrays["global_mean"] - original[index])))}
        h.save(manifest_file, manifest)
        computed += 1
        if computed % 25 == 0:
            print(f"records completed: {len(manifest['records'])}/243", flush=True)
    h.require(all(sha(Path(path)) == value for path, value in before.items()), "Source changed during extraction")
    complete = len(manifest["records"]) == 243
    if complete:
        combined = {key: [] for key in shapes}
        for index in range(243):
            path = cache / f"row_{index:04d}.npz"
            h.require(sha(path) == manifest["records"][str(index)]["sha256"], "Final cache hash mismatch")
            with np.load(path, allow_pickle=False) as saved:
                inspect(saved, index)
                for key in shapes:
                    combined[key].append(saved[key].copy())
        np.savez_compressed(folder / "development_features.npz", **{k: np.stack(v) for k, v in combined.items()},
                            video_ids=videos, patient_ids=people, split=split)
    result = {"status": "complete" if complete else "partial_time_budget", "records_cached": len(manifest["records"]),
              "records_computed": computed, "records_reused": reused, "expected_records": 243,
              "training_records": 203, "validation_records": 40, "development_people": 212,
              "previous_test_records_used": 0, "labels_loaded": False, "score_model_trained": False,
              "source_files_unchanged": True, "fingerprint": fingerprint,
              "max_mean_abs_difference": max(e["max_mean_difference"] for e in manifest["records"].values()),
              "dimensions": {k: v[0] for k, v in shapes.items()}, "elapsed_seconds": time.perf_counter() - started}
    h.save(folder / "cache_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    print(__import__("json").dumps(run(args.session), ensure_ascii=True, indent=2))
