import ast,re
from datetime import datetime,timezone,timedelta
import run as r
v=r.m.load_json(r.HERE/'verification.json');assert v['verified']
ready=r.m.load_json(r.HERE/'input_readiness_snapshot.json');assert ready['exporter_missing_checkpoint_refused']
for name,digest in r.m.load_json(r.HERE/'source_hashes.json').items():assert r.m.digest(r.m.WORKSPACE/name)==digest,name
links=0
for doc in ['report.md','CNN_EXPORT.md']:
    content=(r.HERE/doc).read_text(encoding='utf-8')
    for name in re.findall(r'\]\(([^)]+)\)',content):assert (r.HERE/name).is_file(),name;links+=1
for path in r.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
r.js('execution_state.json',dict(status='CPU_calibration_complete_CNN_pending_train_predictions',
    updated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),report_links_checked=links,
    CPU_calibration_verified=True,CNN_calibration_completed=False,exporter_prepared=True,
    exporter_actual_inference_verified=False,paper_input_snapshot_checked=True,GPU_used=False))
paths=[p for p in r.HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
manifest={p.relative_to(r.HERE).as_posix():r.m.digest(p) for p in sorted(paths)}
r.js('artifact_manifest.json',manifest)
assert all(r.m.digest(r.HERE/name)==digest for name,digest in manifest.items())
print(dict(artifacts=len(manifest),report_links=links,CPU_verified=True,CNN_pending=True))
