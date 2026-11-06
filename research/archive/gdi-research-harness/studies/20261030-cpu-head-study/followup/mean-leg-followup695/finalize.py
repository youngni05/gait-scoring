"""Final source preservation, report links, and immutable inventory checks."""
import re,ast
from datetime import datetime,timezone,timedelta
import run as r
v=r.m.load_json(r.HERE/'verification.json');assert v['verified']
assert r.CFG['ridge_alphas']==r.p.CFG['ridge_alphas']
assert r.CFG['pls_components']==r.p.CFG['pls_components']
sources=r.m.load_json(r.HERE/'source_hashes.json')
for name,digest in sources.items():assert r.m.digest(r.m.WORKSPACE/name)==digest,name
for path in r.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
links=re.findall(r'\]\(([^)]+)\)',(r.HERE/'report.md').read_text(encoding='utf-8'))
for name in links:assert (r.HERE/name).is_file(),name
assert 'mean-leg-followup695/report.md' in (r.HERE.parents[1]/'STATUS.md').read_text(encoding='utf-8')
r.js('execution_state.json',dict(status='completed_and_verified',completed_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
    report_links_checked=len(links),figure_visually_reviewed=True,sources_unchanged=len(sources),GPU_used=False,
    independent_final_evaluation=False,methods=len(r.METHODS),training_tasks=len(r.TASKS)))
paths=[p for p in r.HERE.rglob('*') if p.is_file() and 'mplconfig' not in p.parts and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
manifest={p.relative_to(r.HERE).as_posix():r.m.digest(p) for p in sorted(paths)}
r.js('artifact_manifest.json',manifest)
assert all(r.m.digest(r.HERE/name)==digest for name,digest in manifest.items())
print(dict(artifacts=len(manifest),report_links_checked=len(links),sources_unchanged=len(sources),methods=len(r.METHODS)))
