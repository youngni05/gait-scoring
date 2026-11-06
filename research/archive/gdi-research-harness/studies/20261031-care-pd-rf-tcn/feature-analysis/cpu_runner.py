"""Run feature ablations in a 12-CPU Slurm allocation; never reserve a GPU."""
import os,sys,json,pathlib,subprocess,tempfile,zipfile,concurrent.futures,hashlib
HERE=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/abr/RESEARCH_USER/care-pd-rf-tcn-20261031')
assert os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_CPUS_PER_TASK'])>=12
with tempfile.TemporaryDirectory(prefix='care-rf-features-') as tmp:
    wheels=sorted((BASE/'wheels').glob('*.whl'));assert len(wheels)==4
    for w in wheels:
        with zipfile.ZipFile(w) as z:z.extractall(tmp)
    env=dict(os.environ,PYTHONPATH=tmp,PYTHONDONTWRITEBYTECODE='1',CARE_FEATURE_BASE=str(BASE),CARE_FEATURE_RESULTS=str(BASE/'runs/main'))
    def run(task):
        name,fold=task
        with (HERE/('log-'+name+'-'+str(fold)+'.txt')).open('w') as log:
            p=subprocess.run([sys.executable,'-u',str(HERE/'study.py'),name,'--fold',str(fold)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError((task,p.returncode))
        print('COMPLETE',name,fold,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(run,[(v,f) for v in ['major3d','motion3d','bend3d'] for f in range(6)]))
    subprocess.run([sys.executable,str(HERE/'study.py'),'summarize'],env=env,check=True)
    (HERE/'complete.json').write_text(json.dumps(dict(complete=True,job=os.environ['SLURM_JOB_ID'],fits=90)))
