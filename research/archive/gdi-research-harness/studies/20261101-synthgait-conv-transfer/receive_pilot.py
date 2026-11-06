"""Receive actual pilot/SHA, independently replay on CPU64, approve this worker only."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,tarfile,shutil,datetime,traceback
ROOT=Path('D:/Projects/D_P');H=ROOT/'gdi-research-harness/studies/20261101-synthgait-conv-transfer'
D=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261101-synthgait-conv-transfer')
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST';SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST];SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr+'\n'+p.stdout)
    return p.stdout
def main():
    registry=json.loads((ROOT/'gdi-research-harness/control/gpu-watch/registry.json').read_text(encoding='utf-8-sig'))
    e=next(x for x in registry['experiments'] if x['id']=='synthgait-conv-transfer-692');assert registry['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a'
    submission=json.loads((H/'submission.json').read_text());assert e['jobs']==submission['jobs'] and e['remote']==submission['remote']
    remote=e['remote'];server_gate=json.loads(command(SSH+['cat '+remote+'/control/server-pilot-passed.json']))
    assert server_gate['passed'] and server_gate['manifest_sha256']==sha(H/'manifest.json')
    folder=D/'received'/('pilot-'+e['jobs'][0]);folder.mkdir(parents=True,exist_ok=True);archive=folder/'pilot-review.tar.gz'
    server_sha=command(SSH+['sha256sum '+remote+'/runs/pilot-review.tar.gz']).split()[0]
    if not archive.exists() or sha(archive)!=server_sha:command(SCP+[HOST+':'+remote+'/runs/pilot-review.tar.gz',str(archive)])
    assert sha(archive)==server_sha
    receipt={'passed':True,'server_local_sha256_matched':True,'remote':remote+'/runs/pilot-review.tar.gz',
        'local':str(archive),'server_sha256':server_sha,'local_sha256':sha(archive),'checked_at':datetime.datetime.now().astimezone().isoformat()}
    save(H/'checks/pilot-receipt.json',receipt)
    work=folder/'work';work.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            assert not member.issym() and not member.islnk()
            dest=(work/member.name).resolve();assert work.resolve() in dest.parents or dest==work.resolve()
        tf.extractall(work,filter='data')
    manifest=json.loads((H/'manifest.json').read_text())
    for name,value in manifest.items():assert sha(H/name)==value;shutil.copy2(H/name,work/name)
    shutil.copy2(H/'manifest.json',work/'manifest.json')
    for name,value in server_gate['checkpoints'].items():assert sha(work/name)==value
    os.environ['GDI_DATA_PATH']=str(ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/data.npz')
    os.environ['EXTERNAL_DATA_PATH']=str(D/'external.npz')
    sys.path.insert(0,str(H));import run as r,stem,review
    r.ROOT=work;r.setup(7,'cpu');r.verify_manifest();data=r.data();c=r.config();seed=c['seed']+1
    external=stem.audit_external(work/'runs/pretrain/pilot'/('seed%d'%seed),c,r,'cpu')
    replays={};initial=[]
    for variant in ['scratch','synth']:
        cc=r.config_for(variant,'pilot');dest=work/'runs/pilot'/variant
        summary=r.load(dest/'pilot_summary.json');assert summary['epochs']==3 and summary['train_people']==416 and summary['validation_people']==138 and not summary['outer_evaluation_used']
        replays[variant]=review.replay(data,cc,0,1,3,dest/'pilot','cpu','pilot_not_performance')
        initial.append(r.load(dest/'pilot/initialization.json'))
    assert initial[0]['initial_conv_sha256']==initial[1]['initial_conv_sha256']
    assert initial[0]['other_state_sha256']==initial[1]['other_state_sha256']
    assert initial[1]['all_other_state_unchanged'] and initial[1]['batchnorm_reset']
    curves=[r.load(work/'runs/pilot'/v/'pilot/curve.json') for v in ['scratch','synth']]
    assert [x['sample_sha256'] for x in curves[0]]==[x['sample_sha256'] for x in curves[1]]
    report={'passed':True,'receipt_sha256':sha(H/'checks/pilot-receipt.json'),'manifest_sha256':server_gate['manifest_sha256'],
        'checkpoints':server_gate['checkpoints'],'external_cpu64':external,'GDI_cpu64':replays,'all6_GDI_epoch_audits':True,
        'initial_conv_matched':True,'all_nonconv_initial_state_matched':True,'BN_reset':True,'person_draws_identical':True,
        'outer_evaluation_used':False,'pretraining_uses_GDI':False,'full_performance_available':False,
        'pilot_statistics':{v:r.load(work/'runs/pilot'/v/'pilot_summary.json') for v in ['scratch','synth']}}
    save(H/'checks/pilot-verification.json',report)
    approval={'passed':True,'manifest_sha256':server_gate['manifest_sha256'],'checkpoints':server_gate['checkpoints'],
        'local_report_sha256':sha(H/'checks/pilot-verification.json'),'receipt_sha256':sha(H/'checks/pilot-receipt.json')}
    target=H/'checks/approve-main-local.json';save(target,approval)
    existing=subprocess.run(SSH+['cat '+remote+'/control/approve-main-local.json'],capture_output=True,text=True,timeout=30)
    if existing.returncode==0:assert json.loads(existing.stdout)==approval
    else:command(SCP+[str(target),HOST+':'+remote+'/control/approve-main-local.json'])
    latest=json.loads((ROOT/'gdi-research-harness/control/gpu-watch/registry.json').read_text(encoding='utf-8-sig'))
    row=next(x for x in latest['experiments'] if x['id']=='synthgait-conv-transfer-692');assert row['jobs']==e['jobs']
    row.update(pilot_review_state='reported',pilot_report_path=str(H/'checks/pilot-verification.json'),
        pilot_report_sha256=sha(H/'checks/pilot-verification.json'),result_note='Local SHA+all6epoch/source normalization/initialization/draws/externalCPU64 and2GDI CPU64 pilot checks passed. Approved matching checkpoint SHAs for fixed full comparison; full results pending.')
    save(ROOT/'gdi-research-harness/control/gpu-watch/registry.json',latest)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (H/'checks').mkdir(exist_ok=True);save(H/'checks/pilot-local-failure.json',{'passed':False,'error':repr(exc),'traceback':traceback.format_exc(),'tolerance_changed':False});raise
