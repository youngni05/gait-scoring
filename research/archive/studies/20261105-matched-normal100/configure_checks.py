"""Adapt independent replay checks to matched arms and Stanford-only calibration."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'20261105-child100-train-only'
def sub(s,a,b):assert a in s,a;return s.replace(a,b)
for file in ['verify_pilot.py','audit_results.py','audit_rf.py']:
    s=(OLD/file).read_text()
    s=sub(s,"BIG=Path('D:/gait-research-data/D_P/child100-train-only-20261105')","import argparse\nparser=argparse.ArgumentParser(add_help=False);parser.add_argument('--arm',choices=['adult','child'],required=True);arm_args,_=parser.parse_known_args()\nBIG=Path('D:/gait-research-data/D_P/matched-normal100-20261105')/arm_args.arm")
    if file!='audit_rf.py':s=sub(s,'ap=argparse.ArgumentParser();','ap=argparse.ArgumentParser(parents=[parser]);')
    s=sub(s,"d['source']=='LIVE-TD'","d['source']!='Stanford'")
    s=s.replace("protocol.json')['data_sha256']","protocol.json')['data_sha256'][arm_args.arm]")
    if file!='audit_rf.py':
        s=sub(s,"d['window_start'],d['canonical_fallback'])","d['window_start'],d['canonical_fallback'],d['record_weight'])")
        s=s.replace("d['y'][tr].min()","d['y'][tr&~child].min()").replace("d['y'][tr].max()","d['y'][tr&~child].max()")
        s=s.replace("y[tr].min()","y[tr&~child].min()").replace("y[tr].max()","y[tr&~child].max()")
        if file=='verify_pilot.py':
            s=sub(s,"coefficient=np.linalg.lstsq(np.column_stack([train_raw.reshape(-1).astype(float),np.ones(train_raw.size)]),d['y'][tr].reshape(-1).astype(float),rcond=None)[0]","calraw=train_raw[~child[tr]];coefficient=np.linalg.lstsq(np.column_stack([calraw.reshape(-1).astype(float),np.ones(calraw.size)]),d['y'][tr&~child].reshape(-1).astype(float),rcond=None)[0]")
            s=sub(s,"HERE/f'pilot-{a.family}-verified.json'","HERE/f'pilot-{arm_args.arm}-{a.family}-verified.json'")
        else:
            s=sub(s,'coef=coefficient(train,y[tr])','coef=coefficient(train[~child[tr]],y[tr&~child])')
            s=sub(s,"cr=coefficient(z['train_raw'],y[tr])","cr=coefficient(z['train_raw'][~child[tr]],y[tr&~child])")
            s=sub(s,'train.std(axis=0)','train[~child[tr]].std(axis=0)')
            s=sub(s,"HERE/'audits'/f'{a.family}","HERE/'audits'/arm_args.arm/f'{a.family}")
    else:
        s=sub(s,"(HERE/'rf-audit.json')","(HERE/('rf-audit-'+arm_args.arm+'.json'))")
        # Prior protocol typo explanations do not describe this correctly frozen protocol.
        s=s[:s.index("out=dict(passed=True")]+"out=dict(passed=True,arm=arm_args.arm,folds=report,protocol_sha256=sha(HERE/'rf-protocol.json'))\n(HERE/('rf-audit-'+arm_args.arm+'.json')).write_text(json.dumps(out,indent=2)+'\\n',encoding='utf-8')\nprint(json.dumps(out,indent=2))\n"
    s=s.replace("RUNS=BIG/'received/data/runs'","RUNS=BIG.parent/'received/data'/arm_args.arm/'runs'")
    s=s.replace("BIG/'received/data/runs/mixed'","BIG.parent/'received/data'/arm_args.arm/'runs/mixed'")
    (HERE/file).write_text(s,encoding='utf-8')
print('Replay and selection auditors generated.')
