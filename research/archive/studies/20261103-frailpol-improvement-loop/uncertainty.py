"""Paired person bootstrap of verified results; never used for model selection."""
from pathlib import Path
import json,hashlib,numpy as np
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interval(a):return np.quantile(a,[.025,.975]).tolist()
def main():
    original=read('D:/gait-research-data/D_P/frailpol-objective-threshold-20261103/heldout.json')
    base=[r for r in original if r['target']=='tug' and r['case']=='TabICLv2/raw/features'];assert len(base)==594
    base=sorted(base,key=lambda r:r['id']);ids=[r['id'] for r in base];y=np.array([r['actual'] for r in base]);bp=np.array([r['prediction'] for r in base]);be=np.abs(bp-y)
    rng=np.random.default_rng(20261104);boot=rng.integers(0,len(y),(5000,len(y)));si=np.flatnonzero(y>=20);sb=si[rng.integers(0,len(si),(5000,len(si)))];assert len(si)==9
    results={};provenance={}
    for rd in ['round01','round02','round03','round04','round05','round06']:
        vp=H/rd/'verification.json'
        if not vp.exists():continue
        assert read(vp)['status']=='passed';p=B/rd/'heldout.json';rr=read(p);provenance[rd]=dict(verification_sha256=sha(vp),heldout_sha256=sha(p))
        for condition,kind in sorted(set((r['condition'],r['kind']) for r in rr)):
            selected=sorted([r for r in rr if r['condition']==condition and r['kind']==kind],key=lambda r:r['id']);assert [r['id'] for r in selected]==ids;assert np.array_equal([r['actual'] for r in selected],y)
            pr=np.array([r['prediction'] for r in selected]);e=np.abs(pr-y);bias=pr-y
            speed_cond=condition.split('-')[0]+'-speed' if rd=='round04' else 'speed'
            speed=sorted([r for r in rr if r['condition']==speed_cond and r['kind']==kind],key=lambda r:r['id']);assert [r['id'] for r in speed]==ids;se=np.abs(np.array([r['prediction'] for r in speed])-y)
            bands={}
            for label,mask in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]:
                bands[label]=dict(n=int(mask.sum()),mae=float(e[mask].mean()),bias=float(bias[mask].mean()),max_absolute_error=float(e[mask].max()))
            results[rd+'/'+condition+'/'+kind]=dict(mae=float(e.mean()),mae_ci95=interval(e[boot].mean(axis=1)),p95=float(np.quantile(e,.95)),p95_ci95=interval(np.quantile(e[boot],.95,axis=1)),slow_mae=float(e[si].mean()),slow_mae_ci95=interval(e[sb].mean(axis=1)),mae_difference_original=float((e-be).mean()),mae_difference_original_ci95=interval((e-be)[boot].mean(axis=1)),mae_difference_same_window_speed=float((e-se).mean()),mae_difference_same_window_speed_ci95=interval((e-se)[boot].mean(axis=1)),bands=bands,goal_met=bool(e.mean()<=.9 and np.quantile(e,.95)<=3 and e[si].mean()<=3))
    out=dict(method='5000 person bootstrap; paired same594IDs for differences; slow subgroup bootstrap9persons separately. Development folds reused; intervals do not account for adaptive round selection or external-domain uncertainty. Negative MAE difference favors current condition. No model selection uses this report.',source_hashes=provenance,results=results)
    (H/'uncertainty.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('Verified rounds summarized:',list(provenance))
if __name__=='__main__':main()


