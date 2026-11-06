"""Predeclared same-person waveform versus matched manual-only comparisons."""
from pathlib import Path
import json,hashlib,time,argparse
import numpy as np
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'norm-aux'
PAIRS=[(c,m) for c in ['full_tail4','norm_tail4','symnorm_tail4'] for m in ['speed_tail4','speed_duration_tail4']]+[('symnorm_tail16','speed_tail16'),('symnorm_aux_tail4','speed_aux_tail4'),('symnorm_aux_tail4','speed_duration_aux_tail4')]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def freeze():
    p=dict(code_sha256=sha(Path(__file__)),created_unix=time.time(),wave_protocol_sha256=sha(R/'protocol.json'),manual_protocol_sha256=sha(H/'manual-controls/protocol.json'),pairs=PAIRS,design='Fixedbeforeviewingcompletewaveouterresults. Same594IDs/labels/outerandinnerpartitions,initialpredictionheads/seed/dropout,optimizer/regressionloss/tailweights andauxsettings. Compareeachwavepipelinewithmatchedspeed-only,secondaryspeed+observedcoredurationwhereavailable. Finalepochselectedinsideeachconditiontrainingpeople;notforcedequalafterlookingattest. No conditionselectionorclinicalprobabilityclaim.',checks='Bothsourcefull325checkpointreviews mustpass. RebuildsourceIDs/y;checkeverycorrespondingfitmanualspeednormalizers/targetnormalizers/auxweights/seed/training/evalIDs. Explicitlinearinterpolatedp95/MAE/slow9MAE andsame5000pairedpersonbootstrap;all9pairsreportedwithbands/largeerrors. Inputprojectiondiffersfull24versusnorm6,so wholepipelineablationnotperfectequalparametercountcausality. Singleinit/reuseddevelopmentfolds/conditionalintervalsnotmultiple-selection-adjusted orclinicalexternal.',expected_pairs=9)
    path=R/'matched-manual-protocol.json'
    if path.exists():
        old=read(path);p['created_unix']=old['created_unix'];assert old==p
    else:save(path,p)
def main():
    freeze();assert read(R/'verification.json')['status']==read(H/'manual-controls/verification.json')['status']=='passed';wave=read(B/'norm-aux/heldout.json');manual=read(B/'manual-controls/heldout.json');original=np.load(B/'data.npz');ids=original['ids'];y=original['y'];assert len(ids)==594 and sum(y>=20)==9;rng=np.random.default_rng(20261106);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))];out={}
    def prediction(rows,c):
        a=[r for r in rows if r['condition']==c];lookup={r['id']:r for r in a};assert len(a)==len(lookup)==594 and set(lookup)==set(ids);assert all(lookup[int(i)]['actual']==actual for i,actual in zip(ids,y));return np.array([lookup[int(i)]['prediction'] for i in ids])
    for w,m in PAIRS:
        for fold in range(5):
            a=read(B/'norm-aux'/w/f'fold{fold}/fold.json');b=read(B/'manual-controls'/m/f'fold{fold}/fold.json')
            for wr,mr in list(zip(a['inner'],b['inner']))+[(a['final'],b['final'])]:
                wc=read(wr['context']);mc=read(mr['context']);assert wc['train_ids']==mc['train_ids'] and wc['eval_ids']==mc['eval_ids'] and wc['seed']==mc['seed'];wp=wc['prep'];mp=mc['prep'];assert wp['ym']==mp['ym'] and wp['ys']==mp['ys'] and wp['aux_positive_weight']==mp['positive_weight'];assert wp['manual_mean'][0]==mp['mean'][0] and wp['manual_std'][0]==mp['std'][0]
                if 'duration' in m:assert wp['manual_mean']==mp['mean'] and wp['manual_std']==mp['std']
                else:assert mp['mean'][1]==0 and mp['std'][1]==1
        p=prediction(wave,w);q=prediction(manual,m);e=np.abs(p-y);ce=np.abs(q-y);diff=e-ce;db=diff[boot].mean(1);pd=np.quantile(e[boot],.95,axis=1)-np.quantile(ce[boot],.95,axis=1);out[w+' vs '+m]=dict(n=594,slow_n=9,wave_mae=float(e.mean()),control_mae=float(ce.mean()),mae_difference=float(diff.mean()),mae_difference_ci95=np.quantile(db,[.025,.975]).tolist(),wave_p95=float(np.quantile(e,.95)),control_p95=float(np.quantile(ce,.95)),p95_difference_ci95=np.quantile(pd,[.025,.975]).tolist(),wave_slow_mae=float(e[slow].mean()),control_slow_mae=float(ce[slow].mean()),slow_mae_difference_ci95=np.quantile(diff[sb].mean(1),[.025,.975]).tolist(),wave_errors_over3=int(sum(e>3)),control_errors_over3=int(sum(ce>3)),bands={label:dict(n=int(mask.sum()),wave_mae=float(e[mask].mean()),control_mae=float(ce[mask].mean()),wave_bias=float((p-y)[mask].mean()),control_bias=float((q-y)[mask].mean())) for label,mask in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]})
    save(R/'matched-manual-comparison.json',dict(status='passed_sources_and_paired_comparison',protocol_sha256=sha(R/'matched-manual-protocol.json'),wave_verification_sha256=sha(R/'verification.json'),manual_verification_sha256=sha(H/'manual-controls/verification.json'),comparisons=out,limits='Conditionalpaireddevelopmentintervals,notclinical/external/multiple-selection-adjusted. Negative difference meanslowererror. Noautomaticmodelpromotion.'));print(json.dumps(out,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');a=ap.parse_args();freeze() if a.freeze_only else main()
