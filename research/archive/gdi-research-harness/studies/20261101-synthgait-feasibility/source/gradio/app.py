"""Gradio interface for GaitXFormer video inference."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# Gradio's shared /tmp/gradio default can belong to another cluster user.
os.environ.setdefault(
    "GRADIO_TEMP_DIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / f"gradio-{os.getuid()}"),
)

import gradio as gr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.inference import (
    GaitXFormerPredictor,
    default_checkpoint,
    format_prediction_rows,
)


@lru_cache(maxsize=1)
def get_predictor() -> GaitXFormerPredictor:
    checkpoint = default_checkpoint()
    if not checkpoint:
        raise RuntimeError(
            "Set GAITXFORMER_CHECKPOINT_FILE before launch."
        )
    return GaitXFormerPredictor(
        checkpoint=checkpoint,
        config=REPO_ROOT / "configs" / "demo.yaml",
        device=os.environ.get("GAITXFORMER_DEVICE"),
        detector_model=os.environ.get("GAITXFORMER_DETECTOR", "yolo11n.pt"),
    )


def predict(video_path: str):
    if not video_path:
        raise gr.Error("Upload a video first.")
    result = get_predictor().predict(video_path)
    return format_prediction_rows(result["predictions"])


demo = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="Walking video", sources=["upload"]),
    outputs=gr.Dataframe(
        headers=["Gait parameter", "Estimate"],
        datatype=["str", "str"],
        interactive=False,
        label="Estimated gait parameters",
    ),
    title="GaitXFormer",
    description=(
        "Upload a video containing one full-body walker. GaitXFormer analyzes "
        "the middle five seconds and estimates six gait parameters."
    ),
    article=(
        "**How the gait parameters are calculated**\n\n"
        "- **Cadence:** the number of steps taken per minute.\n"
        "- **Walking Speed:** the forward distance traveled by the sacrum "
        "divided by the elapsed time.\n"
        "- **Mean Step Length:** the average forward distance between "
        "consecutive foot placements.\n"
        "- **Mean Step Width:** the average side-to-side distance between "
        "consecutive foot placements.\n"
        "- **Mean Stooped Posture:** the average forward-backward offset of "
        "the neck from the sacrum across the clip, divided by leg length.\n"
        "- **Arm Swing:** the average of the left and right wrists' "
        "forward-backward movement ranges relative to the sacrum, divided "
        "by leg length."
    ),
    flagging_mode="never",
    examples=[str(REPO_ROOT / "samples" / "sample1.mp4")],
)


if __name__ == "__main__":
    demo.launch(share=True)
