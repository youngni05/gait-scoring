"""Shared GaitXFormer preprocessing and PyTorch inference for public demos."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch

from data.utils import volume_transforms
from data.utils.affine import generate_image_patch_cv2
import data.utils.transforms as video_transforms
from model.gaitxformer import GaitXFormer
from utils.tools import get_config


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
FEATURE_UNITS = {
    "cadence": "steps/min",
    "walking_speed": "m/s",
    "step_length_mean": "m",
    "step_width_mean": "m",
}
# Stooped posture and arm swing are leg-length-normalized ratios in GaitDiff,
# so they are intentionally absent from FEATURE_UNITS.
FEATURE_LABELS = {
    "cadence": "Cadence",
    "walking_speed": "Walking Speed",
    "step_length_mean": "Mean Step Length",
    "step_width_mean": "Mean Step Width",
    "stoop_posture_mean": "Mean Stooped Posture",
    "arm_swing": "Arm Swing",
}


def format_prediction_rows(
    predictions: dict[str, dict[str, Any]],
    precision: int = 4,
) -> list[list[str]]:
    """Format predictions for human-readable terminal and web tables."""
    rows = []
    for feature_name, prediction in predictions.items():
        label = prediction.get(
            "label",
            FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title()),
        )
        estimate = f"{prediction['value']:.{precision}f}"
        if unit := prediction.get("unit"):
            estimate = f"{estimate} {unit}"
        rows.append([label, estimate])
    return rows


def default_checkpoint() -> str | None:
    return os.environ.get("GAITXFORMER_CHECKPOINT_FILE")


def _read_middle_clip(
    video_path: str | Path,
    clip_seconds: float,
    target_fps: float,
) -> tuple[np.ndarray, dict[str, float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or target_fps)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"Could not determine video length for {video_path}")

    source_duration = frame_count / fps
    selected_duration = min(float(clip_seconds), source_duration)
    start_seconds = max(0.0, (source_duration - selected_duration) / 2.0)
    start_frame = max(0, int(round(start_seconds * fps)))
    end_frame = min(frame_count, int(round((start_seconds + selected_duration) * fps)) + 1)

    target_count = max(1, int(round(selected_duration * target_fps)) + 1)
    target_indices = np.linspace(
        start_frame,
        end_frame - 1,
        target_count,
        dtype=int,
    )
    unique_indices, inverse_indices = np.unique(
        target_indices,
        return_inverse=True,
    )

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    selected_frames = []
    next_selected = 0
    for frame_index in range(start_frame, end_frame):
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if (
            next_selected < len(unique_indices)
            and frame_index == unique_indices[next_selected]
        ):
            selected_frames.append(frame_bgr)
            next_selected += 1
    capture.release()
    if len(selected_frames) != len(unique_indices):
        raise ValueError(f"Could not read all selected frames from {video_path}")

    unique_frames = np.stack(selected_frames)
    sampled_frames = unique_frames[inverse_indices]
    metadata = {
        "source_fps": fps,
        "source_duration_seconds": source_duration,
        "clip_start_seconds": start_seconds,
        "clip_end_seconds": start_seconds + selected_duration,
    }
    return sampled_frames, metadata


def _largest_person_boxes(detector, frames_bgr: np.ndarray, device: str) -> np.ndarray:
    results = detector.predict(
        source=list(frames_bgr),
        classes=[0],
        conf=0.25,
        iou=0.5,
        imgsz=640,
        device=device,
        verbose=False,
        stream=False,
    )
    boxes = np.full((len(results), 4), np.nan, dtype=np.float32)
    for index, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confidence = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        area = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(
            0.0,
            xyxy[:, 3] - xyxy[:, 1],
        )
        boxes[index] = xyxy[int(np.argmax(area * np.maximum(confidence, 1e-6)))]
    return boxes


def _trim_and_interpolate_boxes(
    frames_bgr: np.ndarray,
    boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    detected = np.isfinite(boxes).all(axis=1)
    detection_count = int(detected.sum())
    if detection_count == 0:
        raise ValueError(
            "No person was detected in the middle five seconds. Use a video with "
            "one visible, full-body walker."
        )

    detected_indices = np.flatnonzero(detected)
    start, end = int(detected_indices[0]), int(detected_indices[-1]) + 1
    frames_bgr = frames_bgr[start:end]
    boxes = boxes[start:end]
    detected = detected[start:end]
    positions = np.arange(len(boxes))
    known = positions[detected]
    for coordinate in range(4):
        boxes[:, coordinate] = np.interp(
            positions,
            known,
            boxes[detected, coordinate],
        )
    return frames_bgr, boxes, detection_count


def _prepare_model_clip(
    frames_bgr: np.ndarray,
    boxes: np.ndarray,
    height: int,
    n_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_indices = np.linspace(0, len(frames_bgr) - 1, n_frames, dtype=int)
    cropped_frames = []
    for frame_index in sample_indices:
        frame_rgb = cv2.cvtColor(frames_bgr[frame_index], cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = boxes[frame_index]
        center_x = float(x1 + x2) / 2.0
        center_y = float(y1 + y2) / 2.0
        box_size = max(float(x2 - x1), float(y2 - y1))
        box_size = max(2.0, box_size * 1.05)
        cropped, _ = generate_image_patch_cv2(
            frame_rgb,
            center_x,
            center_y,
            box_size,
            box_size,
            height,
            height,
            False,
            1.0,
            0.0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        offset = height // 8
        cropped_frames.append(cropped[:, offset:-offset])

    transform = video_transforms.Compose(
        [
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ]
    )
    clip = transform(np.asarray(cropped_frames)).unsqueeze(0)
    valid = torch.ones((1, n_frames), dtype=torch.bool)
    return clip, valid


class GaitXFormerPredictor:
    """Load the detector and checkpoint once, then predict multiple videos."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: str | Path = REPO_ROOT / "configs" / "demo.yaml",
        device: str | None = None,
        detector_model: str = "yolo11n.pt",
    ):
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "Demo preprocessing requires ultralytics. Install requirements.txt."
            ) from error

        self.args = get_config(str(config))
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested_device)
        self.detector_device = (
            str(self.device.index if self.device.index is not None else 0)
            if self.device.type == "cuda"
            else "cpu"
        )
        self.detector = YOLO(detector_model)

        self.model = GaitXFormer(
            **self.args.model,
            n_out=len(self.args.data.gait_features),
        ).to(self.device)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        stats_path = Path(self.args.data.gait_stats_path)
        if not stats_path.is_absolute():
            stats_path = REPO_ROOT / stats_path
        self.stats = joblib.load(stats_path)
        self.checkpoint_path = str(checkpoint_path)

    @torch.inference_mode()
    def predict(self, video_path: str | Path) -> dict[str, Any]:
        frames, metadata = _read_middle_clip(
            video_path,
            clip_seconds=float(self.args.data.clip_seconds),
            target_fps=float(self.args.data.target_fps),
        )
        boxes = _largest_person_boxes(self.detector, frames, self.detector_device)
        frames, boxes, detection_count = _trim_and_interpolate_boxes(frames, boxes)
        clip, valid = _prepare_model_clip(
            frames,
            boxes,
            height=int(self.args.data.height),
            n_frames=int(self.args.data.n_frames),
        )
        normalized = self.model(
            clip.to(self.device),
            valid.to(self.device),
        )[0].detach().cpu().numpy()

        predictions = {}
        for index, feature_name in enumerate(self.args.data.gait_features):
            feature_stats = self.stats[feature_name]
            prediction = {
                "label": FEATURE_LABELS.get(
                    feature_name,
                    feature_name.replace("_", " ").title(),
                ),
                "value": float(
                    normalized[index] * feature_stats["std"]
                    + feature_stats["mean"]
                ),
            }
            if unit := FEATURE_UNITS.get(feature_name):
                prediction["unit"] = unit
            predictions[feature_name] = prediction
        return {
            "video": str(Path(video_path).resolve()),
            "checkpoint": str(Path(self.checkpoint_path).resolve()),
            **metadata,
            "person_detections": detection_count,
            "model_frames": int(self.args.data.n_frames),
            "predictions": predictions,
        }
