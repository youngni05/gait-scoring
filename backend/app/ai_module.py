"""Load the same checked-in analysis contract in the API and the worker."""
from pathlib import Path
import sys

module_dir = Path(__file__).resolve().parents[2] / "ai-module"
sys.path.insert(0, str(module_dir))

from adapter import analyze_coords, to_request
from history_explanation import compare_and_explain
from prepare_assets import POSE_MODEL_SHA256, verify
from schema import LIMITS, RequestError, parse_json

__all__ = ["LIMITS", "RequestError", "analyze_coords", "compare_and_explain", "parse_json", "to_request"]
