"""Load the selected numerical functions from the checked Stanford sources."""
import ast
import itertools
import json

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d


LEFT = [28, 29, 26, 27, 24, 25, 38, 39, 50, 52, 54, 56]
RIGHT = [22, 23, 20, 21, 18, 19, 44, 45, 51, 53, 55, 57]
REQUIRED = [14, 11, 13, 10, 12, 9, 19, 22]


def source_functions(sources):
    tree = ast.parse((sources / "video_process_utils.py").read_text(encoding="utf-8"))
    # Preserve the original preparation's spelling-only NumPy 2 adaptation.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "np" and node.attr == "NaN"):
            node.attr = "nan"
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    namespace = dict(np=np, interpolate=interpolate, gaussian_filter1d=gaussian_filter1d,
                     itertools=itertools)
    exec(compile(ast.Module(body=body, type_ignores=[]), "pinned_official_utils", "exec"), namespace)
    notebook = json.loads((sources / "process_raw_videos.ipynb").read_text(encoding="utf-8"))
    code = next("".join(cell["source"]) for cell in notebook["cells"]
                if "".join(cell.get("source", [])).startswith("def process_video_and_add_cols"))
    exec(compile(code, "pinned_official_features", "exec"), namespace)
    return namespace
