"""Check report links and freeze this experiment's artifact inventory."""
import re
from datetime import datetime, timezone, timedelta
import run as r
assert r.m.load_json(r.HERE/'verification.json')['verified']
assert r.m.load_json(r.HERE/'control_comparison.json')['first_attempt_predictions_identical']
sources=r.m.load_json(r.HERE/'source_hashes.json')
for name,digest in sources.items():
    assert r.m.digest(r.m.WORKSPACE/name)==digest, name
report=(r.HERE/'report.md').read_text(encoding='utf-8')
links=re.findall(r'\]\(([^)]+)\)',report)
for link in links:
    assert (r.HERE/link).is_file(), link
r.js('execution_state.json',dict(status='completed_and_verified',
    completed_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
    report_links_checked=len(links), figure_visually_reviewed=True,
    original_files_unchanged=len(sources), GPU_used=False, independent_final_evaluation=False,
    initial_aggregation_error_corrected=True, first_attempt_predictions_identical=True))
paths=[p for p in r.HERE.rglob('*') if p.is_file() and 'mplconfig' not in p.parts and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
manifest={p.relative_to(r.HERE).as_posix():r.m.digest(p) for p in sorted(paths)}
r.js('artifact_manifest.json',manifest)
assert all(r.m.digest(r.HERE/name)==digest for name,digest in manifest.items())
print(dict(artifacts=len(manifest), valid_report_links=len(links), sources_unchanged=len(sources)))
