"""CPU-only Slurm worker pool; existing wheel environment, no GPU request."""
import os,sys,json,pathlib,subprocess,tempfile,zipfile,concurrent.futures,hashlib,time
HERE=pathlib.Path(__file__).resolve().parent;BASE=pathlib.Path('/abr/RESEARCH_USER/care-pd-rf-tcn-20261031');mode=sys.argv[1];pilot=mode=='pilot';start=time.time()
assert mode in ['pilot','main'] and os.environ.get('SLURM_JOB_ID')
assert int(os.environ['SLURM_CPUS_PER_TASK'])>=(2 if pilot else 12)
out=HERE/mode;out.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='care-rf2s-') as tmp:
    wheels=sorted((BASE/'wheels').glob('*.whl'));assert len(wheels)==4
    for w in wheels:
        with zipfile.ZipFile(w) as z:z.extractall(tmp)
    env=dict(os.environ,PYTHONPATH=tmp,PYTHONDONTWRITEBYTECODE='1',CARE_FEATURE_BASE=str(BASE),RF2_OUTPUT=str(out))
    if pilot:env['RF2_PILOT']='1'
    def run(task):
        name,fold=task
        with (out/('log-'+name+'-'+str(fold)+'.txt')).open('w') as log:
            p=subprocess.run([sys.executable,'-u',str(HERE/'experiment.py'),'train',name,str(fold)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError((task,p.returncode))
        print('COMPLETE',name,fold,flush=True)
    tasks=[('first2',0)] if pilot else [(v,f) for v in ['full','first2','center2'] for f in range(6)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1 if pilot else 6) as pool:list(pool.map(run,tasks))
    n=1 if pilot else 90;assert len(list((out/'fits').glob('*/*.joblib')))==n
    (out/'complete.json').write_text(json.dumps(dict(passed=True,job=os.environ['SLURM_JOB_ID'],final_models=n,inner_fits=2*n,elapsed_seconds=time.time()-start,script_sha256=hashlib.sha256((HERE/'experiment.py').read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),input_sha256=hashlib.sha256((HERE/'input.npz').read_bytes()).hexdigest())))
