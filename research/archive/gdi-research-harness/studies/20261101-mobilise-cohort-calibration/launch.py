from study import *

if __name__=='__main__':
    pilot=json.loads((HERE/'pilot-verification.json').read_text());assert pilot['passed'] and pilot['tasks']==2
    assert pilot['verify_sha256']==sha(HERE/'verify.py') and pilot['analysis_sha256']==sha(HERE/'analyze.py')
    save(HERE/'main-process.json',{'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script':str(Path(__file__).resolve()),'study_sha256':sha(HERE/'study.py'),'workers':4})
    try:
        run('main',4)
    except BaseException as exc:
        save(HERE/'main-process-exit.json',{'pid':os.getpid(),'exit_code':1,'error_type':type(exc).__name__,'message':str(exc),'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
    else:
        save(HERE/'main-process-exit.json',{'pid':os.getpid(),'exit_code':0,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
