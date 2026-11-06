"""Seal immutable runtime files only before submission."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
assert not (H/'submission.json').exists()
names=['run.py','model.py','windows.py','balanced.py','audit.py','review.py','review_results.py','verify.py','worker.py','protocol.json','training.json','reference_metrics.json','input_contract.json','rank_aux.py','ranking.py','ranksim.py','RANKSIM_LICENSE','source-provenance.json','source_contract.py','runtime_gate.py']
(H/'manifest.json').write_text(json.dumps({n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in names},indent=2)+'\n',encoding='utf-8')
print(hashlib.sha256((H/'manifest.json').read_bytes()).hexdigest())
