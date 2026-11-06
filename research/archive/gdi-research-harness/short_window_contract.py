"""Proposed raw-pose preparation contract; not wired into existing experiments."""
import ast
from pathlib import Path
import numpy as np
import harness as h

ROOT = Path(__file__).resolve().parent
PREPARATION = ROOT.parent / "tools/stanford-sample-check/prepare_ridge_data.py"
CONVERTER = ROOT.parent / "outputs/motionbert-smoke-20261015/assets/mmpose_inference_3d.py"


def source_functions():
    prep = ast.parse(PREPARATION.read_text(encoding="utf-8"))
    adapter = ast.parse(CONVERTER.read_text(encoding="utf-8"))
    normalize = next(n for n in ast.walk(prep) if isinstance(n, ast.FunctionDef) and n.name == "normalize")
    convert = next(n for n in adapter.body if isinstance(n, ast.FunctionDef) and n.name == "convert_keypoint_definition")
    mapping = next(n for n in ast.walk(prep) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "body25_to_coco" for t in n.targets))
    namespace = {"np": np}
    exec(compile(ast.Module(body=[normalize, convert], type_ignores=[]), "original_functions_only", "exec"), namespace)
    return namespace["normalize"], namespace["convert_keypoint_definition"], ast.literal_eval(mapping.value)


def prepare_candidate(poses, frame_ids, length=124):
    """Input already passed the original unique-left-view person selection."""
    h.require(type(length) is int and 1 <= length <= 243, "Unsupported candidate length")
    h.require(poses.ndim == 3 and poses.shape[1:] == (25, 3) and len(poses) == len(frame_ids), "Raw-pose shape mismatch")
    h.require(np.isfinite(poses).all() and frame_ids.ndim == 1 and frame_ids.dtype.kind in "iu", "Invalid numeric input")
    h.require((np.diff(frame_ids) > 0).all(), "Frame IDs must increase; do not silently reorder")
    observed = (poses[:, [9, 10, 11, 12, 13, 14], 2] > 0).all(axis=1)
    best = run = best_start = start = 0
    for i, valid in enumerate(observed):
        if valid:
            if run == 0 or frame_ids[i] != frame_ids[i - 1] + 1:
                run, start = 1, i
            else:
                run += 1
        else:
            run = 0
        if run > best:
            best, best_start = run, start
    h.require(best >= length, "Insufficient continuous observed frames; no padding or interpolation")
    normalize, convert, mapping = source_functions()
    selected = poses[best_start:best_start + length].astype(np.float32)
    h36m = convert(selected[:, mapping], "coco", "h36m")
    window = normalize(h36m)
    h.require(np.array_equal(window[..., 2], h36m[..., 2]), "Normalization would change confidence")
    h.require(window.shape == (length, 17, 3) and np.isfinite(window).all(), "Invalid prepared output")
    return {"input": window, "frame_ids": frame_ids[best_start:best_start + length].copy(),
            "longest_observed_run": best, "start_index": best_start}
