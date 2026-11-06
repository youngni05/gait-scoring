"""Reconstruct only three pinned training records from existing ZIP members."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import zlib

import harness as h
from temporal_readiness import ROOT, DATA, sha
from short_window_contract import prepare_candidate, PREPARATION, CONVERTER

PRIOR = ROOT / "control/desktop-sessions/20261026T061858Z-3624bb"


def run(session):
    started = time.perf_counter()
    import numpy as np
    folder = (ROOT / "control/desktop-sessions" / session).resolve()
    h.require((ROOT / "control/desktop-sessions").resolve() in folder.parents, "Invalid session")
    h.require((folder / "plan.md").is_file(), "Missing prior plan")
    h.require(not (folder / "raw_reconstruction_result.json").exists(), "Reconstruction already saved; inspect evidence instead")
    tools_path = ROOT.parent / "tools/stanford-sample-check"
    sys.path.insert(0, str(tools_path))
    import inspect_archive as arc
    archive = arc.ARCHIVE
    stat_before = archive.stat()
    stat_signature = {"size": stat_before.st_size, "mtime_ns": stat_before.st_mtime_ns}
    dev = ROOT / "jobs/development-baseline/runs/20261025T044558572569Z-71adb0"
    packet = dev / "development_packet.npz"
    dev_receipt = h.load(dev / "receipt.json")
    h.require(dev_receipt["status"] == "completed" and sha(dev / "result.json") == dev_receipt["result_sha256"], "Invalid development receipt")
    h.require(sha(packet) == h.load(dev / "result.json")["artifacts"][0]["sha256"], "Packet changed")
    with np.load(packet, allow_pickle=False) as table:
        videos, people, split = (table[k].copy() for k in ("video_ids", "patient_ids", "split"))
    train = np.flatnonzero(split == "train")
    h.require(len(train) == 203 and len(set(people[train])) == 175, "Wrong training partition")
    chosen = sorted(train, key=lambda i: str(videos[i]))[:3]
    selected_ids = [str(videos[i]) for i in chosen]
    prior_receipt = h.load(PRIOR / "receipt.json")
    h.require(prior_receipt["status"] == "completed" and sha(PRIOR / "response.json") == prior_receipt["response_sha256"], "Invalid previous readiness receipt")
    for name in ("short_training_probes.npz", "short_sources.json"):
        item = next(a for a in prior_receipt["artifacts"] if a["path"] == (PRIOR / name).relative_to(ROOT).as_posix())
        h.require(sha(PRIOR / name) == item["sha256"], "Previous readiness artifact changed")
    previous_rows = h.load(PRIOR / "short_sources.json")["selected_training_records"]
    h.require(selected_ids == [r["video_id"] for r in previous_rows], "Selected records differ from prior probes")
    index_file = ROOT.parent / "outputs/stanford-sample-check-20261016/archive-index.json"
    index = h.load(index_file)
    selected_folders = {}
    for video in selected_ids:
        # Exact original linking rule, no normalization guesses or clinical tables.
        matches = [name for name in index["video_folders"] if len(name.split("/")) >= 2 and name.split("/")[1].removesuffix("-processed") == video]
        h.require(len(matches) == 1, "Raw folder linkage is ambiguous or missing; do not extract")
        selected_folders[matches[0]] = video
    tracked = [packet, dev / "receipt.json", dev / "result.json", index_file, tools_path / "inspect_archive.py",
               tools_path / "run.py", PREPARATION, CONVERTER, ROOT / "short_window_contract.py", Path(__file__),
               PRIOR / "receipt.json", PRIOR / "response.json", PRIOR / "short_training_probes.npz", PRIOR / "short_sources.json"]
    windows = [DATA / "windows" / (v + ".npz") for v in selected_ids]
    tracked += windows
    before = {str(p): sha(p) for p in tracked}
    h.save(folder / "raw_linkage.json", {"selected_training_video_ids": selected_ids,
            "folder_to_video": selected_folders, "link_rule": "second folder component removesuffix('-processed') equals exact pinned video ID",
            "labels_or_predictions_opened": False, "source_sha256": before, "archive_stat_before": stat_signature})
    members = {name: [] for name in selected_folders}
    closed = set()
    previous_folder = None
    with archive.open("rb", buffering=4 * 1024 * 1024) as stream:
        count, directory_size, offset = arc.directory_info(stream)
        h.require(offset + directory_size <= stat_before.st_size, "ZIP index outside archive")
        stream.seek(offset)
        for scanned in range(count):
            h.require(time.perf_counter() - started < 300, "Bounded index scan exceeded five minutes; preserve partial linkage")
            header = stream.read(46)
            h.require(header[:4] == b"PK\x01\x02", "Invalid central header")
            values = struct.unpack("<4s6H3L5H2L", header)
            name = stream.read(values[10]).decode("utf-8")
            stream.seek(values[11] + values[12], 1)
            group = "/".join(name.split("/")[:2])
            if previous_folder in members and group != previous_folder:
                closed.add(previous_folder)
            previous_folder = group
            if len(closed) == len(members):
                break
            if group in members and name.endswith(".json"):
                h.require(not values[3] & 1 and values[4] in (0, 8), "Unsupported/encrypted selected member")
                h.require(values[9] <= 2 * 1024 * 1024 and values[8] <= 2 * 1024 * 1024, "Selected JSON size limit")
                members[group].append({"name": name, "crc32": values[7], "size": values[9], "compressed_size": values[8],
                                       "offset": values[16], "method": values[4]})
                h.require(len(members[group]) <= 5000, "Selected record frame limit")
    for group, items in members.items():
        h.require(len(items) == index["video_folders"][group]["json_files"] and len(items) > 0, "Selected index count mismatch")
        expected_first = index["video_folders"][group]["first_member"]
        for key in ("name", "offset", "crc32", "size", "compressed_size", "method"):
            h.require(items[0][key] == expected_first[key], "Cached first-member index mismatch")
    h.save(folder / "selected_zip_members.json", {"archive": str(archive), "members": members,
             "index_entries_scanned": scanned + 1, "full_archive_hash_verified": False})
    with np.load(PRIOR / "short_training_probes.npz", allow_pickle=False) as old:
        old_crop, old_renormalized = old["crop_inputs"].copy(), old["normalized_inputs"].copy()
    sample_folder = folder / "raw_samples"
    sample_folder.mkdir(exist_ok=True)
    checks, member_proof = [], []
    for sample_number, (video, old_window_path) in enumerate(zip(selected_ids, windows), 1):
        group = next(g for g, v in selected_folders.items() if v == video)
        poses, ids, frame_proof = [], [], []
        exclusions = Counter()
        with archive.open("rb") as data:
            for member in members[group]:
                h.require(time.perf_counter() - started < 600, "Bounded raw reconstruction exceeded ten minutes")
                data.seek(member["offset"])
                header = data.read(30)
                h.require(header[:4] == b"PK\x03\x04", "Invalid local header")
                name_len, extra_len = struct.unpack_from("<HH", header, 26)
                h.require(data.read(name_len).decode("utf-8") == member["name"], "Local member name mismatch")
                data.seek(extra_len, 1)
                compressed = data.read(member["compressed_size"])
                if member["method"] == 8:
                    decoder = zlib.decompressobj(-15)
                    raw = decoder.decompress(compressed, member["size"] + 1)
                    h.require(decoder.eof and not decoder.unconsumed_tail, "Invalid bounded deflate stream")
                else:
                    raw = compressed
                h.require(len(raw) == member["size"] and zlib.crc32(raw) & 0xFFFFFFFF == member["crc32"], "Member size/CRC mismatch")
                obj = json.loads(raw)
                candidates = []
                for person in obj["people"]:
                    points = np.asarray(person["pose_keypoints_2d"], dtype=np.float32)
                    if points.shape != (75,) or not np.isfinite(points).all():
                        exclusions["invalid_person_array"] += 1
                        continue
                    points = points.reshape(25, 3)
                    if points[:, 0].max() < 320:
                        candidates.append(points)
                if len(candidates) == 1:
                    poses.append(candidates[0])
                    ids.append(int(member["name"].rsplit("_", 2)[-2]))
                else:
                    exclusions["no_unique_left_view_person"] += 1
                frame_proof.append({"name": member["name"], "raw_sha256": hashlib.sha256(raw).hexdigest(), "crc32": member["crc32"]})
        h.require(poses, "No uniquely selected person frames")
        poses, ids = np.stack(poses), np.asarray(ids)
        long = prepare_candidate(poses, ids, length=243)
        short = prepare_candidate(poses, ids, length=124)
        with np.load(old_window_path, allow_pickle=False) as old_window:
            original, original_ids = old_window["input"].copy(), old_window["frame_ids"].copy()
        np.testing.assert_array_equal(long["frame_ids"], original_ids)
        np.testing.assert_allclose(long["input"], original, rtol=1e-5, atol=2e-6)
        np.testing.assert_array_equal(long["input"][..., 2], original[..., 2])
        np.testing.assert_array_equal(short["frame_ids"], original_ids[:124])
        np.testing.assert_array_equal(short["input"][..., 2], old_renormalized[sample_number - 1, ..., 2])
        current = short["input"]
        derived = old_renormalized[sample_number - 1]
        observed = current[..., 2] > 0
        coordinate_delta = np.abs(current[..., :2] - derived[..., :2])
        output = sample_folder / f"sample_{sample_number:02d}.npz"
        np.savez_compressed(output, raw_body25=poses, raw_frame_ids=ids, reconstructed243=long["input"],
                            raw124=current, selected243_frame_ids=long["frame_ids"], selected124_frame_ids=short["frame_ids"])
        with np.load(output, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["raw124"], current)
            np.testing.assert_array_equal(saved["reconstructed243"], long["input"])
        checks.append({"sample_number": sample_number, "raw_json_members_checked": len(frame_proof),
                "uniquely_selected_frames": len(ids), "longest_observed_run": long["longest_observed_run"],
                "frame_exclusions": dict(exclusions), "reconstructed243_max_abs_difference": float(np.max(np.abs(long["input"] - original))),
                "raw124_vs_crop_only_max_coordinate_difference": float(np.max(np.abs(current[..., :2] - old_crop[sample_number - 1, ..., :2]))),
                "raw124_vs_renormalized_max_coordinate_difference": float(coordinate_delta.max()),
                "observed_joint_max_coordinate_difference": float(coordinate_delta[observed].max()),
                "unobserved_joint_max_coordinate_difference": float(coordinate_delta[~observed].max()) if (~observed).any() else 0.0,
                "unobserved_joint_positions": int((~observed).sum()), "confidence_preserved": True})
        member_proof.append({"sample_number": sample_number, "video_id": video, "members": frame_proof})
        h.save(folder / "raw_reconstruction_progress.json", {"completed_samples": checks})
    h.require(all(sha(Path(p)) == value for p, value in before.items()), "Source code/cache changed during reconstruction")
    stat_after = archive.stat()
    h.require((stat_after.st_size, stat_after.st_mtime_ns) == (stat_before.st_size, stat_before.st_mtime_ns), "Archive stat changed")
    h.save(folder / "raw_member_proof.json", {"records": member_proof, "archive_stat": stat_signature,
              "integrity_scope": "selected member names, sizes, CRC and extracted-content SHA; whole ZIP hash not rechecked"})
    result = {"status": "passed", "training_records_reconstructed": 3, "training_people": len(set(people[chosen])),
              "total_selected_json_members_checked": sum(c["raw_json_members_checked"] for c in checks),
              "all_original243_reproduced": True, "all_raw124_frames_match_original_prefix": True,
              "confidence_values_preserved": True, "samples": checks,
              "labels_or_predictions_opened": False, "original_test38_pose_content_opened": False,
              "new_people_added": 0, "score_model_trained": False, "source_files_unchanged": True,
              "archive_stat_unchanged": True, "whole_zip_cryptographic_hash_rechecked": False,
              "elapsed_seconds": time.perf_counter() - started}
    h.save(folder / "raw_reconstruction_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.session), ensure_ascii=True, indent=2))
    except Exception as error:
        import traceback
        h.save(ROOT / "control/desktop-sessions" / args.session / "raw_reconstruction_failure.json", {"error_type": type(error).__name__, "details": traceback.format_exc()})
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "details": "saved locally; no automatic retry"}))
        raise SystemExit(1)
