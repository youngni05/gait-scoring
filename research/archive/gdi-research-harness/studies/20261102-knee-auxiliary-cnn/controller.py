"""Read-only status, pilot closeout gate, and final independent CPU review."""
from pathlib import Path, PurePosixPath
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tarfile
import numpy as np
from deploy import HERE, ROOT, BIG, REMOTE, AUTH, SCP, ssh, sha, read, save

sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
import run as r
import review


def status():
    code='''from pathlib import Path
import json,subprocess
root=Path(%r)
def read(p):return json.loads(p.read_text()) if p.exists() else None
result=dict(workers={p.stem:read(p) for p in (root/'control').glob('worker-*.json')},
            pilot_ready=read(root/'control/pilot_ready.json'),
            main_gate=read(root/'control/approve-main.json'),
            summary_complete=read(root/'runs/summary_complete.json'),
            queue=subprocess.check_output(['squeue','-u','RESEARCH_USER','-o','%%.18i %%.24j %%.2t %%.10M %%.10L %%R'],text=True))
for job in ['33809','33810']:
    p=root/'logs'/('worker-'+job+'.log')
    if p.exists():result['tail_'+job]=p.read_text(errors='replace')[-3500:]
print(json.dumps(result))
'''%REMOTE
    return json.loads(ssh(code))


def development_data():
    base=ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/data.npz'
    assert sha(base)==r.config()['data_sha256']
    with np.load(base,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    d['x']=r.b.WindowSet(d['x'],d.pop('window_record'),len(d['y']),d.pop('window_start'),d.pop('canonical_fallback'))
    r.b.validate_data(d)
    with np.load(HERE/'auxiliary-targets.npz') as z:
        for key in ['person_id','video_id','fold']:assert np.array_equal(z[key],d[key])
        d['aux']=z['aux'].copy()
    r.verify_manifest()
    return d


def receive(name,digest,phase):
    dest=BIG/'received'/phase
    dest.mkdir(parents=True,exist_ok=True)
    archive=dest/Path(name).name
    if not archive.exists() or sha(archive)!=digest:
        subprocess.run([SCP,*AUTH,'-P','10000','RESEARCH_USER@PRIVATE_HOST:'+REMOTE+'/'+name,str(archive)],check=True)
    assert sha(archive)==digest,'Server/local archive SHA256 mismatch'
    extracted=dest/'extracted'
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            path=PurePosixPath(m.name)
            assert not path.is_absolute() and '..' not in path.parts and not m.issym() and not m.islnk()
            assert (extracted/Path(*path.parts)).resolve().is_relative_to(extracted.resolve())
        tf.extractall(extracted,filter='data')
    assert sha(extracted/'manifest.json')==sha(HERE/'manifest.json')
    return archive,extracted


def pilot():
    s=status()
    ready=s['pilot_ready']
    assert ready and ready['passed'],'GPU pilot is not ready'
    assert ready['manifest_sha256']==sha(HERE/'manifest.json')
    archive,received=receive('runs/pilot-review.tar.gz',ready['archive_sha256'],'pilot')
    d=development_data()
    r.b.setup(20261102,'cpu')
    proofs={}
    for tag in ['weight-0p0','weight-1p0']:
        proofs[tag]=review.replay(d,received/'runs/pilot'/tag,'cpu')
        assert proofs[tag]['passed']
    local=dict(passed=True,manifest_sha256=sha(HERE/'manifest.json'),archive_sha256=sha(archive),
               original_gpu_ready=ready,replays=proofs,pilot_summary=read(received/'runs/pilot/pilot_summary.json'),
               checked_at=datetime.datetime.now().isoformat(),reserved_holdout_used=False,optimizer_steps=0)
    save(HERE/'checks/local-pilot-review.json',local)
    gate=dict(manifest_sha256=sha(HERE/'manifest.json'),local_pilot_review_passed=True,
              local_review_sha256=sha(HERE/'checks/local-pilot-review.json'),
              pilot_archive_sha256=sha(archive),reviewed_at=datetime.datetime.now().isoformat())
    code='''from pathlib import Path
import json,hashlib
root=Path(%r);gate=json.loads(%r)
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==gate['manifest_sha256']
assert json.loads((root/'control/pilot_ready.json').read_text())['archive_sha256']==gate['pilot_archive_sha256']
path=root/'control/approve-main.json'
if path.exists():assert json.loads(path.read_text())['manifest_sha256']==gate['manifest_sha256']
else:path.write_text(json.dumps(gate,indent=2)+'\\n')
print(json.dumps(dict(approved=True,gate=gate)))
'''%(REMOTE,json.dumps(gate))
    print(ssh(code))


def closeout():
    s=status()
    marker=s['summary_complete']
    assert marker,'Main summary/archive is not complete'
    archive,received=receive('runs/main-review.tar.gz',marker['archive_sha256'],'main')
    d=development_data()
    r.b.setup(20261102,'cpu')
    out=received/'runs/main'
    original=read(out/'summary.json')
    assert sha(out/'summary.json')==marker['summary_sha256']
    # All 90 fit histories and all saved epoch predictions are rechecked locally.
    paths=sorted(out.glob('*/*/*/inner*/latest.pt'))+sorted(out.glob('*/*/refit/latest.pt'))
    assert len(paths)==90
    audits={str(p.parent.relative_to(out)):review.replay(d,p.parent,'cpu',inference=False) for p in paths}
    reinference={str(p.parent.relative_to(out)):review.replay(d,p.parent,'cpu') for p in sorted(out.glob('*/*/refit/latest.pt'))}
    assert len(reinference)==10
    for variant in ['gdi_only','knee_auxiliary']:
        for outer in range(5):
            dest=out/variant/('fold%d'%outer)
            selection=read(dest/'selection.json')
            candidates=[]
            for weight in ([0.0] if variant=='gdi_only' else r.config()['auxiliary_weights']):
                histories=[read(dest/r.tag(weight)/('inner%d'%i)/'curve.json') for i in range(5) if i!=outer]
                epoch,curve=r.b.select_epoch(histories)
                candidates.append(dict(weight=weight,epoch=epoch,score=curve[epoch-1],inner_curve=curve))
            assert selection['chosen']==r.select(candidates) and selection['candidates']==candidates
            selected=selection['chosen']
            ident=read(dest/'refit/identity.json')
            assert ident['auxiliary_weight']==selected['weight'] and ident['epochs']==selected['epoch']
            done=read(dest/'complete.json')
            assert done['selection_sha256']==sha(dest/'selection.json')
            assert done['checkpoint_sha256']==sha(dest/'refit/latest.pt')
            with np.load(dest/'predictions.npz') as z:
                np.testing.assert_allclose(z['predictions'],r.b.corrected(z['predictions_raw'],z['calibration']),atol=1e-10,rtol=0)
    predictions=r.collect(d,out)
    # Independent grouping and per-bootstrap-draw calculation, not run.summarize().
    ids=sorted(set(d['person_id'].tolist()))
    errors={name:np.asarray([sum(abs(float(p[i,j])-float(d['y'][i,j]))
                                for i in range(len(p)) if d['person_id'][i]==person for j in [0,1])
                              /(2*sum(d['person_id']==person)) for person in ids]) for name,p in predictions.items()}
    for name,error in errors.items():assert abs(float(error.mean())-original['metrics'][name]['person_equal_mae'])<1e-10
    delta=errors['knee_auxiliary']-errors['gdi_only']
    rng=np.random.default_rng(r.config()['uncertainty']['seed'])
    samples=[float(np.mean(delta[rng.integers(0,len(ids),len(ids))])) for _ in range(10000)]
    ci=np.percentile(samples,[2.5,97.5])
    np.testing.assert_allclose(ci,original['comparison']['paired_person_bootstrap95'],atol=1e-10,rtol=0)
    assert abs(float(delta.mean())-original['comparison']['auxiliary_minus_gdi_only'])<1e-10
    report=dict(passed=True,archive_sha256=sha(archive),audited_fits=90,
                audited_epochs=sum(v['epochs'] for v in audits.values()),final_model_reinferences=10,
                audits=audits,reinference=reinference,person_mae_and_bootstrap_independently_recomputed=True,
                inner_selection_recomputed=True,reserved_holdout_used=False,real_optimizer_steps=0)
    save(HERE/'checks/final-verification.json',report)
    save(HERE/'results.json',original)
    print(json.dumps(dict(passed=True,metrics=original['metrics'],comparison=original['comparison']),ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['status','pilot','closeout'])
    p.add_argument('--compact',action='store_true')
    a=p.parse_args()
    if a.command=='status':
        s=status();save(HERE/'checks/latest-status.json',s)
        if a.compact:
            print(json.dumps(dict(workers={k:{n:v.get(n) for n in ['job_id','phase','hostname','active_variant','active_fold','error']} for k,v in s['workers'].items()},
                                  pilot_ready=bool(s['pilot_ready']),main_gate=bool(s['main_gate']),summary_complete=bool(s['summary_complete']),
                                  queue=s['queue'],log_tail={k:v.splitlines()[-2:] for k,v in s.items() if k.startswith('tail_')}),indent=2))
        else:print(json.dumps(s,indent=2))
    elif a.command=='pilot':pilot()
    else:closeout()


if __name__=='__main__':main()
