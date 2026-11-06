from pathlib import Path
import time,sys,json,subprocess,os
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');R=H/'norm-aux'
(R/'matched-review-process.json').write_text(json.dumps(dict(pid=os.getpid(),status='waiting_for_both_source_reviews')),encoding='utf8')
while True:
    ready=True
    for root in [R,H/'manual-controls']:
        try:v=json.loads((root/'verification.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            ready=False;p=root/'review-process.json'
            if p.exists() and json.loads(p.read_text()).get('status')=='failed':raise RuntimeError('Sourcefullreviewfailed;pairedanalysisnotrun')
            continue
        assert v['status']=='passed','Sourceverificationfailed;do notpromotecomparison'
    if ready:break
    time.sleep(10)
r=subprocess.run([sys.executable,str(H/'compare_matched_manual.py')],stdout=(R/'matched-review-log.txt').open('w'),stderr=subprocess.STDOUT)
(R/'matched-review-process.json').write_text(json.dumps(dict(pid=os.getpid(),status='verified' if r.returncode==0 else 'failed',exit_code=r.returncode)),encoding='utf8');raise SystemExit(r.returncode)
