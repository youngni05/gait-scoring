"""Run GaitXFormer on the middle five seconds of a video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.inference import (
    GaitXFormerPredictor,
    default_checkpoint,
    format_prediction_rows,
)


def format_result_table(result: dict) -> str:
    """Render gait estimates as a compact terminal table."""
    headers = ["Gait parameter", "Estimate"]
    rows = format_prediction_rows(result["predictions"])
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def row(values: list[str]) -> str:
        cells = [f" {value:<{width}} " for value, width in zip(values, widths)]
        return "│" + "│".join(cells) + "│"

    lines = [
        border("┌", "┬", "┐"),
        row(headers),
        border("├", "┼", "┤"),
        *(row(values) for values in rows),
        border("└", "┴", "┘"),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--checkpoint",
        default=default_checkpoint(),
        help=(
            "Checkpoint file (defaults to $GAITXFORMER_CHECKPOINT_FILE)"
        ),
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument(
        "--detector-model",
        default="yolo11n.pt",
        help="Ultralytics person-detector weights or model name",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for saving the full result as JSON",
    )
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error(
            "set GAITXFORMER_CHECKPOINT_FILE or pass --checkpoint FILE"
        )
    return args


def main() -> None:
    args = parse_args()
    predictor = GaitXFormerPredictor(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        detector_model=args.detector_model,
    )
    result = predictor.predict(args.video)
    print(f"\nGaitXFormer estimates — {Path(result['video']).name}")
    print(
        "Analyzed segment: "
        f"{result['clip_start_seconds']:.2f}–{result['clip_end_seconds']:.2f} s\n"
    )
    print(format_result_table(result))
    if args.json_output:
        output = json.dumps(result, indent=2, sort_keys=True)
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"\nSaved machine-readable results to {output_path}")


if __name__ == "__main__":
    main()
