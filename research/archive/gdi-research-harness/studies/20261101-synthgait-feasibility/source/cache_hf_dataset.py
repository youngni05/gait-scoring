"""Build a six-target FoundationGait cache from downloaded HF WebDataset shards."""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import tarfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.gait_features import GAIT_FEATURES as FEATURE_NAMES
from data.video_preprocessing import GaitVideoPreprocessor
from utils.inference import _largest_person_boxes, _trim_and_interpolate_boxes


SHARD_SPLITS = {"train": "train", "val": "validation"}
_WORKER_PROCESSOR = None
_WORKER_DETECTOR = None
_WORKER_DEVICE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=os.environ.get("SYNTHGAIT_ROOT"),
        required=os.environ.get("SYNTHGAIT_ROOT") is None,
        help="Downloaded HF dataset root (defaults to $SYNTHGAIT_ROOT).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=os.environ.get("SYNTHGAIT_CACHE"),
        required=os.environ.get("SYNTHGAIT_CACHE") is None,
        help="Training-cache root (defaults to $SYNTHGAIT_CACHE).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SHARD_SPLITS),
        default=tuple(SHARD_SPLITS),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--detector-model",
        default="yolo11n.pt",
        help="Ultralytics person-detector weights or model name.",
    )
    parser.add_argument(
        "--detector-device",
        default="cpu",
        help="Ultralytics device, such as cpu or 0. Use one worker with a GPU.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _make_processor() -> GaitVideoPreprocessor:
    return GaitVideoPreprocessor(
        height=256,
        n_frames=32,
        max_frames=81,
        initial_fps=16,
    )


def _init_worker(detector_model: str, detector_device: str) -> None:
    global _WORKER_DETECTOR, _WORKER_DEVICE, _WORKER_PROCESSOR
    from ultralytics import YOLO

    _WORKER_PROCESSOR = _make_processor()
    _WORKER_DETECTOR = YOLO(detector_model)
    _WORKER_DEVICE = detector_device


def _processor_and_detector(detector_model: str, detector_device: str):
    if _WORKER_PROCESSOR is None:
        _init_worker(detector_model, detector_device)
    return _WORKER_PROCESSOR, _WORKER_DETECTOR, _WORKER_DEVICE


def _box_masks(frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    height, width = frames.shape[1:3]
    masks = np.zeros((len(frames), height, width), dtype=np.float32)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        left = max(0, min(width - 1, int(np.floor(x1))))
        top = max(0, min(height - 1, int(np.floor(y1))))
        right = max(left + 1, min(width, int(np.ceil(x2))))
        bottom = max(top + 1, min(height, int(np.ceil(y2))))
        masks[index, top:bottom, left:right] = 1.0
    return masks


def _cache_sample(
    processor: GaitVideoPreprocessor,
    detector,
    detector_device: str,
    members: dict[str, bytes],
    output_path: Path,
    overwrite: bool,
) -> int:
    if output_path.exists() and not overwrite:
        with np.load(output_path, allow_pickle=False) as cached:
            return int(np.asarray(cached["frames_valid"]).sum())

    required = {"mp4", "npz"}
    missing = required - set(members)
    if missing:
        raise RuntimeError(f"{output_path.stem}: missing shard members {sorted(missing)}")

    video_frames, frame_indices = processor.load_video(io.BytesIO(members["mp4"]))
    del frame_indices
    frames_bgr = video_frames[..., ::-1].copy()
    boxes = _largest_person_boxes(detector, frames_bgr, detector_device)
    video_frames, boxes, _ = _trim_and_interpolate_boxes(video_frames, boxes)
    mask_frames = _box_masks(video_frames, boxes)
    if len(video_frames) > processor.max_frames:
        start_index = (len(video_frames) - processor.max_frames) // 2
        video_frames = video_frames[
            start_index : start_index + processor.max_frames
        ]
        mask_frames = mask_frames[
            start_index : start_index + processor.max_frames
        ]
    valid_indices = processor.frame_indices[
        processor.frame_indices < len(video_frames)
    ]
    video_frames = video_frames[valid_indices]
    mask_frames = mask_frames[valid_indices]
    video_frames, mask_frames = processor.crop_resize(video_frames, mask_frames)
    video_frames, mask_frames = processor.zero_pad_frames(video_frames, mask_frames)
    frames_valid = (mask_frames.sum(axis=(1, 2)) > 0).astype(np.int64)
    video = processor.transforms(video_frames).numpy()

    with np.load(io.BytesIO(members["npz"]), allow_pickle=False) as annotation:
        features = np.asarray(annotation["features"], dtype=np.float32)
    if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
        raise RuntimeError(
            f"{output_path.stem}: expected {len(FEATURE_NAMES)} finite gait features, "
            f"found {features.shape}"
        )

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("wb") as stream:
        np.savez(
            stream,
            video=video,
            gait_features=features,
            feature_names=np.asarray(FEATURE_NAMES),
            frames_valid=frames_valid,
        )
    os.replace(temporary_path, output_path)
    return int(frames_valid.sum())


def _member_parts(name: str) -> tuple[str, str] | None:
    basename = Path(name).name
    for suffix in ("mp4", "npz", "json"):
        ending = f".{suffix}"
        if basename.endswith(ending):
            return basename[: -len(ending)], suffix
    return None


def _cache_shard(
    task: tuple[str, str, bool, str, str],
) -> list[tuple[str, int, str]]:
    shard_name, output_dir_name, overwrite, detector_model, detector_device = task
    output_dir = Path(output_dir_name)
    processor, detector, detector_device = _processor_and_detector(
        detector_model,
        detector_device,
    )
    records = []
    current_id: str | None = None
    current_members: dict[str, bytes] = {}

    def flush() -> None:
        nonlocal current_id, current_members
        if current_id is None:
            return
        output_name = f"{current_id}.npz"
        valid_count = _cache_sample(
            processor,
            detector,
            detector_device,
            current_members,
            output_dir / output_name,
            overwrite,
        )
        records.append((output_name, valid_count, f"{current_id}.mp4"))
        current_id = None
        current_members = {}

    with tarfile.open(shard_name, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = _member_parts(member.name)
            if parts is None:
                continue
            sample_id, suffix = parts
            if current_id is not None and sample_id != current_id:
                flush()
            current_id = sample_id
            if suffix != "json":
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"Could not read {member.name} from {shard_name}")
                current_members[suffix] = stream.read()
        flush()
    return records


def _write_json(path: Path, value) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, path)


def cache_split(opts: argparse.Namespace, cache_split: str) -> None:
    release_split = SHARD_SPLITS[cache_split]
    dataset_dir = opts.dataset_dir.expanduser().resolve()
    output_dir = opts.output_dir.expanduser().resolve() / cache_split
    shards = sorted((dataset_dir / "data").glob(f"{release_split}-*.tar"))
    if not shards:
        raise FileNotFoundError(
            f"No {release_split} shards found in {dataset_dir / 'data'}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(path),
            str(output_dir),
            opts.overwrite,
            opts.detector_model,
            opts.detector_device,
        )
        for path in shards
    ]
    records: list[tuple[str, int, str]] = []
    worker_count = max(1, min(opts.num_workers, len(tasks)))
    if worker_count == 1:
        iterator = map(_cache_shard, tasks)
        for shard_records in tqdm(iterator, total=len(tasks), desc=f"Caching {cache_split}"):
            records.extend(shard_records)
    else:
        with mp.get_context("spawn").Pool(
            worker_count,
            initializer=_init_worker,
            initargs=(opts.detector_model, opts.detector_device),
        ) as pool:
            iterator = pool.imap_unordered(_cache_shard, tasks)
            for shard_records in tqdm(
                iterator,
                total=len(tasks),
                desc=f"Caching {cache_split}",
            ):
                records.extend(shard_records)

    records.sort()
    names = [name for name, _, _ in records]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate sample IDs found in {release_split} shards")
    _write_json(
        output_dir / "valid_frame_counts.json",
        {name: count for name, count, _ in records},
    )
    _write_json(
        output_dir / "npz_to_video_map.json",
        {name: video for name, _, video in records},
    )
    print(f"[INFO] {cache_split}: {len(records):,} clips in {output_dir}")


def main() -> None:
    opts = parse_args()
    if opts.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if opts.detector_device != "cpu" and opts.num_workers != 1:
        raise ValueError("Use --num-workers 1 when --detector-device is a GPU")
    for split in opts.splits:
        cache_split(opts, split)


if __name__ == "__main__":
    main()
