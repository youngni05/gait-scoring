"""Resume unchanged work functions with threads after a stalled Windows process pool."""
from study import *
from concurrent.futures import ThreadPoolExecutor

if __name__=='__main__':
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for f,h in protocol['sources'].items():assert sha(f)==h
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
    initialize();start=time.time()
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in tasks]),1):
            tag,seconds=f.result()
            if i%10==0:print(i,len(tasks),tag,seconds,round(time.time()-start),flush=True)
    save(HERE/'main-completion.json',{'tasks':300,'seconds':time.time()-start,'code_sha256':sha(HERE/'study.py'),'runner_sha256':sha(__file__),'recovery':'unchanged work functions, 4 threads; prior completed tasks reused'})
