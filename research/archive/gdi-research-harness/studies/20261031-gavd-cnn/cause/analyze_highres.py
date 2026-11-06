"""Same-record, common-window pose-resolution sensitivity; no preferred-score selection."""
import csv,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
base=rows(DATA/'runs/confidence_primary/predictions/window_predictions.csv');new=rows(DATA/'cause/highres_predictions/window_predictions.csv')
lookup={(r['record_id'],r['fold'],r['frame_first']):r for r in base};details=[]
for r in new:
    key=(r['record_id'],r['fold'],r['frame_first'])
    if key not in lookup:continue
    b=lookup[key];details.append(dict(record_id=key[0],fold=int(key[1]),frame_first=int(key[2]),baseline=(float(b['left_GDI'])+float(b['right_GDI']))/2,highres=(float(r['left_GDI'])+float(r['right_GDI']))/2))
review=rows(HERE/'reviewed_highres.csv');summary=[]
for r in review:
    vv=[v for v in details if v['record_id']==r['record_id']];assert vv and {v['fold'] for v in vv}==set(range(5))
    b=float(np.mean([v['baseline'] for v in vv]));n=float(np.mean([v['highres'] for v in vv]));summary.append(dict(record_id=r['record_id'],video_id=r['video_id'],label=r['label'],view=r['view'],baseline_score=b,highres_score=n,delta=n-b,common_windows=len(vv)//5))
for name,table in [('highres_scores.csv',summary),('highres_common_windows.csv',details)]:
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
mapping={r['record_id']:r for r in summary};pairs=[]
for p in rows(STUDY/'matched_pairs_shift0.csv'):
    n=mapping[p['normal_record']];a=mapping[p['abnormal_record']];pairs.append(dict(view=p['view'],fps=p['normal_fps'],baseline_difference=n['baseline_score']-a['baseline_score'],highres_difference=n['highres_score']-a['highres_score']))
result=dict(records=summary,matched_pairs=pairs,mean_absolute_score_change=float(np.mean([abs(r['delta']) for r in summary])),max_absolute_score_change=max(abs(r['delta']) for r in summary),interpretation='Same frames, weights, target rule and common windows. Different network input size; neither extraction is a verified anatomical reference.')
(HERE/'highres_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
