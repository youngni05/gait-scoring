"""Frozen CNN responses to the same public child inputs; no GDI ground truth."""
from pathlib import Path
import json,hashlib,sys,argparse,collections
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;ROOT=HERE.parents[3]
BUNDLE=ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
sys.path.insert(0,str(BUNDLE))
from infer import verify_bundle,validate_input,configure,predict_records,aggregate
from model import GdiCNN,channels

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def candidate_predictions(mode,d,received=None):
    audit=STUDY/mode/'analysis/comparison.json'
    if not audit.exists():raise ValueError('Full candidate audit is required before child inference: '+mode)
    records=[];windows=[];provenance=[]
    config=read(STUDY/mode/'bundle/protocol.json')
    for f in range(5):
        folder=(received or STUDY/mode/'received')/'results'/('fold%d'%f)
        complete=read(folder/'complete.json');path=folder/'refit/latest.pt'
        assert sha(path)==complete['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=True)
        m=GdiCNN(config['dropout']);m.load_state_dict(ck['model'],strict=True);m.eval()
        raw=[]
        with torch.inference_mode():
            for start in range(0,len(d['x']),128):
                x=channels(torch.from_numpy(d['x'][start:start+128])).reshape(-1,12,124)
                raw.append((m(x)[:,0]*ck['span']+ck['lo']).reshape(-1,2).numpy())
        raw=np.concatenate(raw);slope,intercept=complete['calibration']
        records.append(aggregate(raw,d['window_record'],len(d['record_id']))*slope+intercept)
        windows.append(raw.astype(np.float64)*slope+intercept)
        provenance.append(dict(fold=f,checkpoint_sha256=sha(path),complete_sha256=sha(folder/'complete.json')))
    return np.stack(records),np.stack(windows),dict(candidate_audit_sha256=sha(audit),folds=provenance)

def summarize(records,windows,d,quality):
    plan=read(HERE/'age_matching_plan.json');people=np.unique(d['person_id'])
    ages=plan['person_age'];counts=collections.Counter(int(ages[p]) for p in people)
    target=plan['development_year_counts'];represented=sum(target[str(a)] for a in counts)
    weights=np.array([target[str(int(ages[p]))]/represented/counts[int(ages[p])] for p in people])
    scores=np.array([records.mean(axis=0)[d['person_id']==p].mean(axis=0) for p in people])
    movement={ (r['record'],w['start']):w['movement_fraction'] for r in quality for w in r['scheduled_windows'] }
    coverage=np.array([movement[(str(d['record_id'][owner]),int(k))] for owner,k in zip(d['window_record'],d['window_start'])])
    diagnostics={}
    for name,mask in [('mostly_stationary',coverage<.2),('mixed',(.2<=coverage)&(coverage<.8)),('mostly_moving',coverage>=.8)]:
        values=windows.mean(axis=0)[mask];pids=d['person_id'][d['window_record'][mask]]
        perperson=np.array([values[pids==p].mean() for p in np.unique(pids)])
        diagnostics[name]=dict(windows=int(mask.sum()),people=len(perperson),unweighted_person_mean=float(perperson.mean()) if len(perperson) else None)
    return dict(people=len(people),records=len(d['record_id']),windows=len(d['x']),unweighted_person_mean=float(scores.mean()),age_adjusted_person_mean=float(np.sum(scores.mean(axis=1)*weights)),age_adjusted_left=float(np.sum(scores[:,0]*weights)),age_adjusted_right=float(np.sum(scores[:,1]*weights)),person_range=[float(scores.mean(axis=1).min()),float(scores.mean(axis=1).max())],represented_target_people=represented,target_people=sum(target.values()),full_age5to12_coverage=len(counts)==8,unrepresented_years=[int(a) for a in target if int(a) not in counts],effective_n=float(1/np.sum(weights**2)),participants=[dict(person=str(p),age=ages[p],left=float(s[0]),right=float(s[1]),mean=float(s.mean()),weight=float(w)) for p,s,w in zip(people,scores,weights)],movement_diagnostics=diagnostics)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--quality',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode',choices=['baseline','high90_x3','rare_sqrt_cap4'],default='baseline');ap.add_argument('--received',type=Path);args=ap.parse_args()
    configure();verify_bundle();d=load(args.prepared);validate_input(d)
    if args.mode=='baseline':
        rows=[predict_records(d['x'],d['window_record'],len(d['record_id']),f) for f in range(5)]
        records=np.stack([r[1] for r in rows]);windows=np.stack([r[2] for r in rows]);provenance=dict(bundle_manifest_sha256=sha(BUNDLE/'manifest.json'))
    else:records,windows,provenance=candidate_predictions(args.mode,d,args.received)
    summary=summarize(records,windows,d,read(args.quality))
    result=dict(mode=args.mode,scope='Exploratory response to automatically selected public child video clips, including standing; target QA and continuous-gait validity must be reviewed separately',actual_GDI_available=False,input_sha256=sha(args.prepared),quality_sha256=sha(args.quality),age_plan_sha256=sha(HERE/'age_matching_plan.json'),provenance=provenance,summary=summary)
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/'predictions.npz',predictions=records,window_predictions=windows,person_id=d['person_id'],record_id=d['record_id'])
    (args.out/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='participants'},indent=2))
if __name__=='__main__':main()
