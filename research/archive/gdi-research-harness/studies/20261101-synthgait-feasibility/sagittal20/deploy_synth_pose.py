"""Upload isolated inference payload, submit held, record actual job before release."""
from pathlib import Path
import subprocess,json,hashlib,datetime
ROOT=Path('D:/Projects/D_P');HERE=ROOT/'gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST';SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST];SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def command(args,timeout=60):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr+'\n'+p.stdout)
    return p.stdout
def main():
    p=json.loads((HERE/'pose-transfer.json').read_text());remote=p['remote']
    assert not (HERE/'pose-submission.json').exists(),'Existing submission must be inspected before any resubmit'
    command(SSH+['test ! -e '+remote+' && test ! -e '+remote+'.tar'])
    command(SCP+[p['archive'],HOST+':'+remote+'.tar'],timeout=180)
    actual=command(SSH+['sha256sum '+remote+'.tar']).split()[0];assert actual==p['sha256']
    command(SSH+['mkdir '+remote+' && tar -xf '+remote+'.tar -C '+remote+' && mkdir '+remote+'/logs && ln -s /abr/RESEARCH_USER/gavd-cnn-20261031/network '+remote+'/network'])
    digest=command(SSH+['sha256sum '+remote+'/network/network.json '+remote+'/network/network_weights.npz'])
    for name,expected in p['network'].items():assert expected in digest
    response=command(SSH+['cd '+remote+' && sbatch --hold --parsable pose.slurm']).strip();job=response.split(';')[0];assert job.isdigit(),response
    submission={'job_id':job,'remote':remote,'held':True,'archive_sha256':actual,'inference_only':True,'optimizer_steps':0,
                'expected_frames':p['expected_frames'],'expected_records':20,'submitted_at':datetime.datetime.now().astimezone().isoformat()}
    (HERE/'pose-submission.json').write_text(json.dumps(submission,indent=2),encoding='utf-8')
    reg=ROOT/'gdi-research-harness/control/gpu-watch/registry.json';r=json.loads(reg.read_text(encoding='utf-8-sig'))
    assert r['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a' and r['automation_id']=='gdi-2'
    assert not any(e['id']=='synthgait-sagittal20-body25' for e in r['experiments'])
    r['experiments'].append({'id':'synthgait-sagittal20-body25','kind':'data_extraction','local':str(HERE.relative_to(ROOT)).replace('\\','/'),
        'remote':remote,'jobs':[job],'preserve_allocations':[],'expected_fits':0,'expected_outer_folds':0,
        'expected_frames':p['expected_frames'],'expected_records':20,'run_root':'full','curve_glob':'no_training_curves/*',
        'outer_complete_glob':'no_outer_evaluation/*','summary_path':'full/complete.json','marker_path':'full/complete.json',
        'review_state':'pending','reported_digest':None,
        'audit':'Inference only, zero optimizer steps. Read sagittal20 README, pose-transfer/submission and download-manifest. Check Slurm exit0, full/complete20records1231frames, source video and network hashes, 3 official CPU/GPU BODY25 heatmap probes at unchanged1e-4. Retrieve index, chunks and logs to D with server/local SHA receipt, run official OpenPose CPU PAF decoding and fixed 12-feature extraction. Preserve source16FPS and annotation30Hz; all20clips25..81frames shorter than124. No fake repetition, forced FPS, GDI target, training job or whole OOF comparison. Pilot preparation stays blocked until time/input contract resolved. Existing RankSim/ordinal jobs unchanged.'})
    reg.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(submission,indent=2))
if __name__=='__main__':main()
