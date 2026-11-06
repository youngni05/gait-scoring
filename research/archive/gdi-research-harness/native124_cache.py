"""Bounded resumable raw124 feature preparation on the pinned 243 development rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import zlib

import harness as h
from temporal_readiness import ROOT, DATA, MB, sha, EXPECTED_WEIGHT_SHA256
from short_window_contract import prepare_candidate, PREPARATION, CONVERTER

PILOT = ROOT / "control/desktop-sessions/20261026T100042Z-9d41cd"
CACHE = ROOT / "jobs/native124-cache-v1"
DEV = ROOT / "jobs/development-baseline/runs/20261025T044558572569Z-71adb0"


def artifact_verified(receipt, path):
    item = next(a for a in receipt["artifacts"] if a["path"] == path.relative_to(ROOT).as_posix())
    h.require(sha(path) == item["sha256"], "Pilot artifact changed")


def load_context():
    import numpy as np
    receipt = h.load(PILOT / "receipt.json")
    h.require(receipt["status"] == "completed" and sha(PILOT / "response.json") == receipt["response_sha256"], "Invalid pilot completion")
    for name in ("native124_tasks.json", "raw124_features.npz", "raw_linkage.json", "raw_member_proof.json"):
        artifact_verified(receipt, PILOT / name)
    protocol = h.load(PILOT / "native124_tasks.json")
    packet = DEV / "development_packet.npz"
    dev_receipt = h.load(DEV / "receipt.json")
    h.require(dev_receipt["status"] == "completed" and sha(DEV / "result.json") == dev_receipt["result_sha256"], "Invalid development completion")
    h.require(sha(packet) == protocol["source_packet_sha256"] == h.load(DEV / "result.json")["artifacts"][0]["sha256"], "Development packet changed")
    with np.load(packet, allow_pickle=False) as table:
        # Deliberately no access to Y or any original evaluation table.
        videos, people, split = (table[k].copy() for k in ("video_ids", "patient_ids", "split"))
    tasks = protocol["tasks"]
    h.require(len(tasks) == len(set(videos)) == 243 and len(set(people)) == 212, "Wrong cohort")
    h.require(sum(split == "train") == 203 and sum(split == "validation") == 40, "Wrong split")
    h.require(not set(people[split == "train"]) & set(people[split == "validation"]), "Person overlap")
    for t, v, p, s in zip(tasks, videos, people, split):
        h.require((t["video_id"], t["patient_id"], t["split"]) == (str(v), str(p), str(s)), "Task order changed")
    sys.path.insert(0, str(ROOT.parent / "tools/stanford-sample-check"))
    import inspect_archive as arc
    st = arc.ARCHIVE.stat()
    h.require({"size": st.st_size, "mtime_ns": st.st_mtime_ns} == h.load(PILOT / "raw_linkage.json")["archive_stat_before"], "Original archive metadata changed")
    tracked = [Path(__file__), ROOT / "short_window_contract.py", PREPARATION, CONVERTER,
               ROOT.parent / "tools/stanford-sample-check/inspect_archive.py", packet, DEV / "result.json", DEV / "receipt.json",
               PILOT / "receipt.json", PILOT / "response.json", PILOT / "native124_tasks.json", PILOT / "raw124_features.npz",
               PILOT / "raw_linkage.json", PILOT / "raw_member_proof.json", MB / "assets/MB_lite.bin",
               ROOT.parent / "outputs/stanford-sample-check-20261016/archive-index.json"]
    tracked += sorted((MB / "vendor/lib/model").rglob("*.py"))
    source_hashes = {str(p): sha(p) for p in tracked}
    h.require(source_hashes[str(MB / "assets/MB_lite.bin")] == EXPECTED_WEIGHT_SHA256 == protocol["weight_sha256"], "Weight mismatch")
    h.require(source_hashes[str(tracked[15])] == protocol["archive_index_sha256"], "Archive index changed")
    config = {"version": 1, "frames": 124, "pooling": "mean time and joints", "maxlen": 243,
              "threads": 4, "rtol": 1e-5, "atol": 2e-6, "source_sha256": source_hashes,
              "archive_path": str(arc.ARCHIVE), "archive_stat": {"size": st.st_size, "mtime_ns": st.st_mtime_ns},
              "numpy": np.__version__, "no_labels": True, "cohort_records": 243}
    return tasks, videos, people, split, arc, config


def scan_members(arc, tasks):
    start = time.perf_counter()
    groups = {t["raw_folder"]: [] for t in tasks}
    h.require(len(groups) == 243, "Duplicate raw folders")
    with arc.ARCHIVE.open("rb", buffering=4 * 1024 * 1024) as stream:
        count, _, offset = arc.directory_info(stream)
        stream.seek(offset)
        for _ in range(count):
            h.require(time.perf_counter() - start < 300, "Index scan exceeded five minutes")
            header = stream.read(46)
            h.require(header[:4] == b"PK\x01\x02", "Bad ZIP central header")
            v = struct.unpack("<4s6H3L5H2L", header)
            name = stream.read(v[10]).decode("utf-8")
            stream.seek(v[11] + v[12], 1)
            group = "/".join(name.split("/")[:2])
            if group in groups and name.endswith(".json"):
                h.require(not v[3] & 1 and v[4] in (0, 8), "Unsupported selected member")
                h.require(v[9] <= 2 * 1024 * 1024 and v[8] <= 2 * 1024 * 1024, "JSON exceeds bounded read size")
                groups[group].append({"name": name, "crc32": v[7], "size": v[9], "compressed_size": v[8], "offset": v[16], "method": v[4]})
    for task in tasks:
        h.require(0 < len(groups[task["raw_folder"]]) == task["expected_json_members"] <= 20000, "Raw member count mismatch/limit")
    return groups


def read_poses(archive, members):
    import numpy as np
    poses, ids, proof = [], [], []
    with archive.open("rb") as stream:
        for m in members:
            stream.seek(m["offset"])
            header = stream.read(30)
            h.require(header[:4] == b"PK\x03\x04", "Bad ZIP local header")
            nl, el = struct.unpack_from("<HH", header, 26)
            h.require(stream.read(nl).decode("utf-8") == m["name"], "Member name mismatch")
            stream.seek(el, 1)
            compressed = stream.read(m["compressed_size"])
            if m["method"] == 8:
                decoder = zlib.decompressobj(-15)
                raw = decoder.decompress(compressed, m["size"] + 1)
                h.require(decoder.eof and not decoder.unconsumed_tail, "Invalid bounded compressed stream")
            else:
                raw = compressed
            h.require(len(raw) == m["size"] and zlib.crc32(raw) & 0xFFFFFFFF == m["crc32"], "Raw size/CRC mismatch")
            people = []
            for person in json.loads(raw)["people"]:
                points = np.asarray(person["pose_keypoints_2d"], dtype=np.float32)
                if points.shape != (75,) or not np.isfinite(points).all():
                    continue
                points = points.reshape(25, 3)
                if points[:, 0].max() < 320:
                    people.append(points)
            if len(people) == 1:
                poses.append(people[0])
                ids.append(int(m["name"].rsplit("_", 2)[-2]))
            proof.append({"name": m["name"], "crc32": m["crc32"], "raw_sha256": hashlib.sha256(raw).hexdigest()})
    h.require(poses, "No uniquely selected person frames")
    return np.stack(poses), np.asarray(ids), proof


def check_cached(index, entry):
    import numpy as np
    path = CACHE / "records" / f"row_{index:04d}.npz"
    proof = CACHE / "proof" / f"row_{index:04d}.json"
    h.require(sha(path) == entry["sha256"] and sha(proof) == entry["proof_sha256"], "Completed cache fingerprint mismatch")
    with np.load(path, allow_pickle=False) as row:
        h.require(row["input124"].shape == (124, 17, 3) and row["feature"].shape == (512,), "Cached shape mismatch")
        h.require(np.isfinite(row["input124"]).all() and np.isfinite(row["feature"]).all(), "Cached numeric error")
        h.require((row["input124"][:, 1:7, 2] > 0).all() and (np.diff(row["frame_ids124"]) == 1).all(), "Cached observation mismatch")
    return path


def save_record(index, task, short, ids, feature, proof, window_hash, maximum, state, fingerprint):
    import numpy as np
    path = CACHE / "records" / f"row_{index:04d}.npz"
    proof_path = CACHE / "proof" / f"row_{index:04d}.json"
    h.require(not path.exists() and not proof_path.exists(), "Uncommitted record artifacts; inspect before recovery")
    with path.with_suffix(".tmp").open("wb") as out:
        np.savez_compressed(out, input124=short, frame_ids124=ids, feature=feature)
        out.flush()
        os.fsync(out.fileno())
    os.replace(path.with_suffix(".tmp"), path)
    h.save(proof_path, proof)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["feature"], feature)
        np.testing.assert_array_equal(saved["input124"], short)
        np.testing.assert_array_equal(saved["frame_ids124"], ids)
    state["records"][str(index)] = {"sha256": sha(path), "proof_sha256": sha(proof_path),
            "video_id": task["video_id"], "split": task["split"], "reference243_sha256": window_hash,
            "reconstructed243_max_abs_difference": maximum, "config_fingerprint": fingerprint}
    h.save(CACHE / "state.json", state)


def run(session, maximum=12, inspect=False):
    started = time.perf_counter()
    import numpy as np
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents and (folder / "plan.md").is_file(), "Missing valid session plan")
    h.require(type(maximum) is int and 1 <= maximum <= 12, "Batch size must be 1..12")
    CACHE.mkdir(exist_ok=True)
    with h.exclusive(CACHE / "lock"):
        tasks, videos, people, split, arc, config = load_context()
        for path in (MB / "vendor", MB / "deps", DATA / "deps"):
            sys.path.insert(0, str(path))
        import torch
        config["torch"] = torch.__version__
        fingerprint = h.digest(json.dumps(config, sort_keys=True).encode())
        if (CACHE / "config.json").exists():
            h.require(h.load(CACHE / "config.json") == config, "Immutable cache condition changed")
            state = h.load(CACHE / "state.json")
            h.require(state["fingerprint"] == fingerprint, "Cache state fingerprint mismatch")
            h.require(sha(CACHE / "zip_members.json") == state["member_index_sha256"], "Selected member index changed")
            h.require(not state.get("failure"), "Earlier record failed; review before any retry")
        else:
            (CACHE / "records").mkdir(exist_ok=True)
            (CACHE / "proof").mkdir(exist_ok=True)
            members = scan_members(arc, tasks)
            h.save(CACHE / "zip_members.json", members)
            h.save(CACHE / "config.json", config)
            state = {"fingerprint": fingerprint, "member_index_sha256": sha(CACHE / "zip_members.json"), "records": {}}
            h.save(CACHE / "state.json", state)
        for key, entry in state["records"].items():
            index = int(key)
            h.require(entry["config_fingerprint"] == fingerprint and entry["video_id"] == tasks[index]["video_id"], "Cached record linkage changed")
            h.require(sha(DATA / "windows" / (tasks[index]["video_id"] + ".npz")) == entry["reference243_sha256"], "Original cached window changed")
            check_cached(index, entry)
        reused_pilot = 0
        if not inspect:
            receipt = h.load(PILOT / "receipt.json")
            selected_ids = h.load(PILOT / "raw_linkage.json")["selected_training_video_ids"]
            with np.load(PILOT / "raw124_features.npz", allow_pickle=False) as old:
                old_features = old["features_raw124"].copy()
            for i, task in enumerate(tasks):
                if task["video_id"] in selected_ids and str(i) not in state["records"]:
                    n = selected_ids.index(task["video_id"])
                    source = PILOT / "raw_samples" / f"sample_{n + 1:02d}.npz"
                    artifact_verified(receipt, source)
                    h.require(sha(source) == task["pilot_artifact"]["sha256"], "Pilot task artifact mismatch")
                    old_window = DATA / "windows" / (task["video_id"] + ".npz")
                    with np.load(source, allow_pickle=False) as raw, np.load(old_window, allow_pickle=False) as original:
                        np.testing.assert_array_equal(raw["reconstructed243"], original["input"])
                        np.testing.assert_array_equal(raw["selected243_frame_ids"], original["frame_ids"])
                        save_record(i, task, raw["raw124"], raw["selected124_frame_ids"], old_features[n],
                             {"source": "verified prior pilot", "pilot_artifact": task["pilot_artifact"],
                              "raw_member_proof_sha256": config["source_sha256"][str(PILOT / "raw_member_proof.json")]},
                             sha(old_window), 0.0, state, fingerprint)
                    reused_pilot += 1
        pending = [i for i in range(243) if str(i) not in state["records"]]
        computed = 0
        if pending and not inspect:
            from lib.model.DSTformer import DSTformer
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                              num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243, qkv_bias=True, att_fuse=True)
            cp = torch.load(MB / "assets/MB_lite.bin", map_location="cpu", weights_only=True)
            model.load_state_dict({k.removeprefix("module."): v for k, v in cp["model_pos"].items()}, strict=True)
            model.requires_grad_(False)
            model.eval()
            members = h.load(CACHE / "zip_members.json")
            for i in pending[:maximum]:
                task = tasks[i]
                try:
                    h.require(time.perf_counter() - started < 600, "Bounded batch exceeded ten minutes")
                    poses, frame_ids, proof = read_poses(arc.ARCHIVE, members[task["raw_folder"]])
                    long = prepare_candidate(poses, frame_ids, 243)
                    short = prepare_candidate(poses, frame_ids, 124)
                    original_file = DATA / "windows" / (task["video_id"] + ".npz")
                    original_hash = sha(original_file)
                    with np.load(original_file, allow_pickle=False) as original:
                        reference = original["input"].copy()
                        np.testing.assert_array_equal(long["frame_ids"], original["frame_ids"])
                        np.testing.assert_allclose(long["input"], reference, rtol=1e-5, atol=2e-6)
                        np.testing.assert_array_equal(long["input"][..., 2], reference[..., 2])
                        np.testing.assert_array_equal(short["frame_ids"], original["frame_ids"][:124])
                        np.testing.assert_array_equal(short["input"][..., 2], reference[:124, ..., 2])
                    with torch.inference_mode():
                        rep = model.get_representation(torch.from_numpy(short["input"][None]))
                        h.require(tuple(rep.shape) == (1, 124, 17, 512) and torch.isfinite(rep).all().item(), "Invalid model representation")
                        feature = rep.mean(dim=(1, 2)).numpy()[0].copy()
                    h.require(sha(original_file) == original_hash, "Original window changed during batch")
                    save_record(i, task, short["input"], short["frame_ids"], feature,
                           {"source": "selected raw ZIP members", "members": proof}, original_hash,
                           float(np.max(np.abs(long["input"] - reference))), state, fingerprint)
                    computed += 1
                except Exception as error:
                    import traceback
                    failure = {"row_index": i, "error_type": type(error).__name__, "details": traceback.format_exc(), "at": h.now()}
                    h.save(folder / "native124_failure.json", failure)
                    state["failure"] = {"row_index": i, "evidence": str((folder / "native124_failure.json").relative_to(ROOT))}
                    h.save(CACHE / "state.json", state)
                    raise
        for p, expected in config["source_sha256"].items():
            h.require(sha(Path(p)) == expected, "Pinned source changed")
        st = arc.ARCHIVE.stat()
        h.require({"size": st.st_size, "mtime_ns": st.st_mtime_ns} == config["archive_stat"], "Archive metadata changed")
        complete = len(state["records"]) == 243
        if complete and not (CACHE / "development_features124.npz").exists():
            features = []
            for i in range(243):
                path = check_cached(i, state["records"][str(i)])
                with np.load(path, allow_pickle=False) as row:
                    features.append(row["feature"].astype(np.float64))
            np.savez_compressed(CACHE / "development_features124.npz", X=np.stack(features),
                                video_ids=videos, patient_ids=people, split=split)
        result = {"status": "complete" if complete else "partial", "cached_records": len(state["records"]),
                  "pending_records": 243 - len(state["records"]), "newly_computed": computed, "pilot_records_reused": reused_pilot,
                  "labels_loaded": False, "score_model_trained": False, "new_people_added": 0,
                  "source_files_unchanged": True, "fingerprint": fingerprint, "elapsed_seconds": time.perf_counter() - started}
        h.save(folder / f"batch_{len(state['records']):03d}_{'inspect' if inspect else 'run'}.json", result)
        h.save(folder / "native124_state_snapshot.json", state)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--max-records", type=int, default=12)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.session, args.max_records, args.inspect), ensure_ascii=True))
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "details": "preserved locally; no automatic repeated attempt"}))
        raise SystemExit(1)
