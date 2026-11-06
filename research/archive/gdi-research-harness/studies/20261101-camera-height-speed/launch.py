from pathlib import Path
import subprocess,sys,json,datetime,os
HERE=Path(__file__).resolve().parent
def save(name,x):(HERE/name).write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':
    pilot=json.loads((HERE/'pilot-verification.json').read_text());assert pilot['passed'] and pilot['clips']==8
    child=subprocess.Popen([sys.executable,'-X','utf8',str(HERE/'study.py'),'main'])
    save('main-process.json',dict(pid=child.pid,launcher_pid=os.getpid(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),entry='study.py main'))
    code=child.wait();save('main-process-exit.json',dict(pid=child.pid,exit_code=code))
    if code==0:
        for script,args in [('analyze.py',[]),('verify.py',['main'])]:
            code=subprocess.call([sys.executable,'-X','utf8',str(HERE/script)]+args)
            if code:break
    save('pipeline-exit.json',dict(exit_code=code,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()));sys.exit(code)
