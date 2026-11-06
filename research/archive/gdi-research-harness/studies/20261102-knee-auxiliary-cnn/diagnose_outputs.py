"""Describe all frozen outer predictions; do not exclude folds or tune models."""
from pathlib import Path
import json
import hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261102-knee-auxiliary-cnn/received/main/extracted/runs/main')


def main():
    report=dict(optimizer_steps=0,excluded_folds=[],folds=[])
    for fold in range(5):
        row=dict(fold=fold,variants={})
        for name in ['gdi_only','knee_auxiliary']:
            path=OUT/name/('fold%d'%fold)/'predictions.npz'
            with np.load(path,allow_pickle=False) as z:
                p=z['predictions'];raw=z['predictions_raw']
                people=np.unique(z['person_id'])
                error=np.mean([np.abs(p[z['person_id']==person]-z['y'][z['person_id']==person]).mean() for person in people])
                row['variants'][name]=dict(people=len(people),records=len(p),mae=float(error),
                    raw_min=float(raw.min()),raw_max=float(raw.max()),raw_std=float(raw.std()),
                    unique_raw_values=int(len(np.unique(raw))),
                    corrected_min=float(p.min()),corrected_max=float(p.max()),
                    corrected_std=float(p.std()),prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        row['auxiliary_minus_control']=row['variants']['knee_auxiliary']['mae']-row['variants']['gdi_only']['mae']
        report['folds'].append(row)
    (HERE/'fold-diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
