"""Credential-free transport for this experiment; uses registered SSH agent key."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
REMOTE='/abr/RESEARCH_USER/care-pd-rf-tcn-20261031'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
      '-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST'


def command(source, timeout=100):
    p=subprocess.run(['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST,'python3 -'],
        input=source,text=True,encoding='utf8',capture_output=True,timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr+p.stdout)
    return p.stdout


def main():
    a=argparse.ArgumentParser(); a.add_argument('mode'); args=a.parse_args()
    prefix='import pathlib,json,subprocess,hashlib,zipfile\np=pathlib.Path('+repr(REMOTE)+')\n'
    if args.mode=='pilot':
        digest=hashlib.sha256((HERE/'bundle.zip').read_bytes()).hexdigest()
        source=prefix+"z=p.with_suffix('.zip')\nassert hashlib.sha256(z.read_bytes()).hexdigest()=="+repr(digest)+"\nassert not p.exists()\nzipfile.ZipFile(z).extractall(p.parent)\n(p/'logs').mkdir()\nr=subprocess.run(['sbatch','--nodelist=DIS02','--exclude=DIS03,DIS06','--time=00:20:00','submit.slurm','pilot'],cwd=p,capture_output=True,text=True,check=True)\nprint(json.dumps(dict(submission=r.stdout.strip(),bundle_sha256="+repr(digest)+")))\n"
    elif args.mode=='status':
        source=prefix+"q=subprocess.run(['squeue','-h','-u','RESEARCH_USER','-o','%i|%j|%T|%M|%R'],capture_output=True,text=True,check=True).stdout\nfiles={str(f.relative_to(p)):f.read_text() for f in p.glob('runs/**/summary.json')}\ncomplete=[str(f.relative_to(p)) for f in p.glob('runs/main/fold*/seed*/complete.json')]\nlogs={f.name:f.read_text()[-2200:] for f in (p/'logs').glob('*.log')}\nprint(json.dumps(dict(queue=q,complete=complete,summaries=files,logs=logs)))\n"
    elif args.mode=='retry-pilot':
        source=prefix+"r=subprocess.run(['sbatch','--nodelist=DIS02','--exclude=DIS03,DIS06','--time=00:20:00','submit.slurm','pilot'],cwd=p,capture_output=True,text=True,check=True)\nprint(r.stdout)\n"
    elif args.mode=='train':
        source=prefix+"assert json.loads((p/'runs/pilot/summary.json').read_text())['passed']\nassert not (p/'runs/main').exists()\nresults=[]\nfor node,folds in [('DIS01','0,1,2'),('DIS02','3,4,5')]:\n r=subprocess.run(['sbatch','--nodelist='+node,'--exclude=DIS03,DIS06','--time=04:00:00','submit.slurm','train',folds],cwd=p,capture_output=True,text=True,check=True)\n results.append(dict(node=node,folds=folds,submission=r.stdout.strip()))\nprint(json.dumps(results))\n"
    else: raise ValueError(args.mode)
    result=command(source); print(result)
    (HERE/(args.mode+'-remote.json')).write_text(result,encoding='utf8')


if __name__=='__main__':main()
