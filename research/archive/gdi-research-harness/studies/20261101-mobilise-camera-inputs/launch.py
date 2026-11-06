from study import *

if __name__=='__main__':
    pv=json.loads((HERE/'pilot-verification.json').read_text());assert pv['passed'] and pv['tasks']==2 and pv['verify_sha256']==sha(HERE/'verify.py') and pv['analysis_sha256']==sha(HERE/'analyze.py')
    save(HERE/'main-process.json',{'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'study_sha256':sha(HERE/'study.py'),'workers':4})
    try:run('main',4)
    except BaseException as exc:
        save(HERE/'main-process-exit.json',{'pid':os.getpid(),'exit_code':1,'error':str(exc)});raise
    else:save(HERE/'main-process-exit.json',{'pid':os.getpid(),'exit_code':0})
