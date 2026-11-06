"""Seal experiment files before CPU verification and submission."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
names=['run.py','model.py','windows.py','balanced.py','audit.py','review.py','verify.py','worker.py','protocol.json','reference_metrics.json','input_contract.json','rank_aux.py','ranking.py','ranksim.py','RANKSIM_LICENSE','controls.py','select_strength.py','source-provenance.json','review_results.py']
m={n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in names}
(H/'manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print(hashlib.sha256((H/'manifest.json').read_bytes()).hexdigest())
