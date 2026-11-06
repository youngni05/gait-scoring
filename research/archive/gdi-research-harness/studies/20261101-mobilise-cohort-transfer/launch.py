from study import *

if __name__=='__main__':
    v=json.loads((HERE/'pilot-verification.json').read_text());assert v['passed'] and v['tasks']==2
    save(HERE/'main-process.json',{'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script':str(Path(__file__).resolve()),'study_sha256':sha(HERE/'study.py'),'workers':4})
    run('main',4)
