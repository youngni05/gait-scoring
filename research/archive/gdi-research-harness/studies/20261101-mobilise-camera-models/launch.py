from pathlib import Path
import os,sys,json,hashlib,datetime,subprocess
HERE=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
def save(q,value):Path(q).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    pv=json.loads((HERE/'pilot-verification.json').read_text());assert pv['passed'] and pv['tasks']==2 and pv['verify_sha256']==sha(HERE/'verify.py') and pv['analysis_sha256']==sha(HERE/'analyze.py')
    child=subprocess.Popen([sys.executable,'-X','utf8',str(HERE/'study.py'),'main','--workers','4'])
    save(HERE/'main-process.json',{'pid':child.pid,'launcher_pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'study_sha256':sha(HERE/'study.py'),'workers':4,'entry':'direct study.py main'})
    save(HERE/'main-process-exit.json',{'pid':child.pid,'exit_code':None,'state':'running'})
    code=child.wait();save(HERE/'main-process-exit.json',{'pid':child.pid,'exit_code':code,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()});sys.exit(code)
