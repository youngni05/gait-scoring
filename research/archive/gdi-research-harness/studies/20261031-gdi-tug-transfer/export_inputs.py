"""Export model inputs only after explicit target-identity visual review."""
from pathlib import Path
import json,hashlib
import numpy as np,openpyxl,torch
from features import ROOT,OUT,BUNDLE,encode,extract_windows,GdiCNN,channels,sha
HERE=Path(__file__).resolve().parent
def main():
    qa=json.loads((OUT/'quality-final.json').read_text())
    assert qa['reviewed_all_tracks'] and qa['before_cohort_predictions']
    tracks={r['person']:r for r in json.loads((OUT/'track-review.json').read_text())}
    old=np.load(ROOT/'gdi-research-harness/studies/20261030-tug-toronto-timeseries/input.npz')
    table=ROOT/'outputs/gait-public-access-20261015/Toronto/Table 1.xlsx'
    wb=openpyxl.load_workbook(table,read_only=True,data_only=True);rows=list(wb['Baseline Information'].values);wb.close()
    col=next(i for i,v in enumerate(rows[0]) if isinstance(v,str) and v.strip()=='TUG (s)')
    labels={str(int(r[0])):float(r[col]) for r in rows[1:] if type(r[0]) in (float,int)}
    xs=[];ids=[];ys=[];folds=[];excluded=[]
    for person in sorted(tracks,key=int):
        item=tracks[person];p=OUT/'tracks'/item['track_file'];assert sha(p)==item['sha256']
        assert qa['track_sha256'][person]==sha(p)
        if person in qa.get('exclusions',{}):excluded.append({'person':person,'reason':qa['exclusions'][person]});continue
        try:windows=extract_windows(np.load(p)['raw'])
        except ValueError as e:excluded.append({'person':person,'reason':str(e)});continue
        assert windows['x'].shape==(1,2,124,12)
        mask=old['ids']==person;assert mask.any() and np.all(old['y'][mask]==labels[person])
        fold=np.unique(old['folds'][mask]);assert len(fold)==1
        xs.append(windows['x'][0]);ids.append(person);ys.append(labels[person]);folds.append(int(fold[0]))
    assert set(folds)==set(range(5)),'Exclusions broke outer folds; stop for review'
    x=np.stack(xs);base,score,learned,replays=encode(x)
    random=[]
    for seed in range(20261031,20261036):
        torch.manual_seed(seed);m=GdiCNN(.5).eval();seen=[]
        h=m.output.register_forward_pre_hook(lambda layer,args:seen.append(args[0].detach().clone()))
        with torch.inference_mode():m(channels(torch.from_numpy(x)).reshape(-1,12,124))
        h.remove();random.append(seen[0].numpy().reshape(len(x),-1))
    np.savez_compressed(OUT/'prepared.npz',x=x,base=base,gdi_score=score,gdi_features=learned,random_features=np.concatenate(random,axis=1),ids=np.array(ids),y=np.array(ys),folds=np.array(folds))
    result={'people':len(ids),'excluded':excluded,'prepared_sha256':sha(OUT/'prepared.npz'),'clinical_sha256':sha(table),
            'protocol_sha256':sha(HERE/'protocol.json'),'quality_sha256':sha(OUT/'quality-final.json'),'replays':replays,
            'original_fold_assignments_retained':True,'clinical_labels_exact':True,'scope':'Prepared, not yet trained or evaluated'}
    (OUT/'input-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
