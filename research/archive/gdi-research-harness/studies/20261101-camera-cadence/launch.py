from pathlib import Path
import subprocess,sys,os,json,datetime,hashlib
HERE=Path(__file__).resolve().parent
def save(n,x):(HERE/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':
    v=json.loads((HERE/'pilot-verification.json').read_text());assert v['passed'] and v['tasks']==4 and v['estimates']==32 and v['verify_sha256']==hashlib.sha256((HERE/'verify.py').read_bytes()).hexdigest()
    child=subprocess.Popen([sys.executable,'-X','utf8',str(HERE/'study.py'),'main']);save('main-process.json',dict(pid=child.pid,launcher_pid=os.getpid(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),entry='study.py main'));code=child.wait();save('main-process-exit.json',dict(pid=child.pid,exit_code=code))
    if not code:
        for name,args in [('analyze.py',[]),('verify.py',['main'])]:
            code=subprocess.call([sys.executable,'-X','utf8',str(HERE/name)]+args)
            if code:break
    save('pipeline-exit.json',dict(exit_code=code,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()));sys.exit(code)
