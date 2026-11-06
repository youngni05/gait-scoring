"""Read-only reuse of original code and sources; small outputs stay in this study."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDIES = HERE.parent
CPU = STUDIES / '20261030-cpu-head-study/followup'
FUSION = STUDIES / '20261030-gdi-fusion695'
GPU = STUDIES / '20261030-gpu-motionbert-study'
MB = ROOT / 'outputs/motionbert-smoke-20261015'
TORONTO = STUDIES / '20261030-toronto-poma-tug'
TRANSFER = Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
BUNDLE = ROOT / 'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
SOURCES = {}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None:
        assert digest == expected, str(path)
    SOURCES[str(path)] = digest
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def arrays(path):
    source(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save(name, obj):
    p = HERE / name
    p.resolve().relative_to(HERE)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def functions(path, names, namespace=None, flexible_batch=False):
    """Extract existing pure functions, avoiding their training entry points."""
    path = source(path)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = [next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            for name in names]
    if flexible_batch:
        # Source feature math is batch independent; generalize its 1064-row assertion only.
        for function in body:
            for node in ast.walk(function):
                if isinstance(node, ast.Assert):
                    for child in ast.walk(node):
                        if isinstance(child, ast.Constant) and child.value == 1064:
                            child.value = -1064
                    class BatchSize(ast.NodeTransformer):
                        def visit_Constant(self, item):
                            if item.value == -1064:
                                return ast.copy_location(ast.Call(func=ast.Name(id='len', ctx=ast.Load()),
                                    args=[ast.Name(id='x', ctx=ast.Load())], keywords=[]), item)
                            return item
                    BatchSize().visit(node)
    ns = {'np': np, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), ns)
    return ns


def module(path, name):
    source(path)
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def clinical():
    import openpyxl
    path = source(ROOT / 'outputs/gait-public-access-20261015/Toronto/Table 1.xlsx',
                  '5b9d1e4541c045cc25c452eb719e7438f38408d83e1600a582648a62533c6e24')
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb['Baseline Information'].values)
    wb.close()
    cols = {str(v).strip(): i for i, v in enumerate(rows[0]) if v is not None}
    return {str(int(row[0])): [float(row[cols['TUG (s)']]), float(row[cols['POMA-Gait']])]
            for row in rows[1:] if type(row[0]) in (int, float)}
