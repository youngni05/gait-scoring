"""Save record-level scores for the fixed comparison. No new fitting."""
import json

import numpy as np

from analyze import HERE, owner, pred, rows

out = []
for r, row in enumerate(rows):
    ix = np.flatnonzero(owner == r)
    if row['primary'] and len(ix):
        out.append(dict(person=row['person'], member=row['member'], archive=row['archive'], frames=row['frames'],
                        windows=len(ix), starts=row['starts'], baseline=float(pred['baseline'][ix].mean()),
                        added=float(pred['stan'][ix].mean()),
                        left_right_added=[float(v) for v in pred['stan'][ix].mean(0)]))
(HERE / 'trial-scores.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print(len(out))
