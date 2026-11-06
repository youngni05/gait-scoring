"""Read-only progress snapshot; runs on the login node, no model computation."""
import datetime,json,subprocess
from pathlib import Path
root=Path('/abr/RESEARCH_USER/ljmu100-mixed-20261104')
fits=[];completed=[]
for family in ['CNN','STT','bilateral']:
    for outer in range(5):
        folder=root/'data/runs/mixed'/family/('fold%d'%outer)
        if (folder/'complete.json').exists():completed.append({'family':family,'fold':outer})
        for path in sorted(folder.glob('*/curve.json')):
            try:
                h=json.loads(path.read_text());ident=json.loads((path.parent/'identity.json').read_text())
                fits.append({'family':family,'fold':outer,'fit':path.parent.name,'epoch':len(h),'epochs':ident['epochs'],
                             'complete':(path.parent/'fit_complete.json').exists(),
                             'recent_seconds':sum(v['seconds'] for v in h[-10:])/len(h[-10:])})
            except (ValueError,FileNotFoundError):pass
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'sacct':subprocess.check_output(['sacct','-j','35866,35867','--format=JobID,State,ExitCode,Elapsed,NodeList','--parsable2','--noheader']).decode(),
                  'completed_folds':completed,'completed_fits':sum(v['complete'] for v in fits),'fits':fits}))
