"""Preserve the first pilot; freeze a separate numeric-OLS correction revision."""
from pathlib import Path
import hashlib,json,shutil
H=Path(__file__).resolve().parent;R=H/'revision2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    proof=json.loads((H/'checks/stable-calibration-cross-platform.json').read_text());assert proof['passed'] and proof['maximum_difference']==0
    R.mkdir(exist_ok=True)
    for arm in (0,1):
        old=H/('seed%d'%arm);new=R/('seed%d'%arm);new.mkdir(exist_ok=True)
        manifest=json.loads((old/'manifest.json').read_text())
        for name,digest in manifest.items():
            assert sha(old/name)==digest;shutil.copyfile(old/name,new/name)
        source=(old/'run.py').read_text(encoding='utf-8')
        a=source.index('def calibrate(');b=source.index('\n\ndef corrected(',a)
        source=source[:a]+"def calibrate(train_predictions,train_labels):\n    from stable_calibration import stable_calibrate\n    return stable_calibrate(train_predictions,train_labels)\n"+source[b:]
        (new/'run.py').write_text(source,encoding='utf-8',newline='\n')
        shutil.copyfile(H/'stable_calibration.py',new/'stable_calibration.py')
        p=json.loads((old/'provenance.json').read_text());p.update(original_revision_manifest_sha256=sha(old/'manifest.json'),numeric_correction='Same training-only pooled OLS, centered deterministic fsum; original tolerances unchanged')
        save(new/'provenance.json',p)
        save(new/'manifest.json',{p.name:sha(p) for p in sorted(new.iterdir()) if p.is_file() and p.name!='manifest.json'})
    shutil.copyfile(H/'protocol.json',R/'protocol.json')
    source=(H/'manage.py').read_text(encoding='utf-8')
    source=source.replace("WORK=H.parents[2]","WORK=Path('D:/Projects/D_P')")
    source=source.replace('/abr/RESEARCH_USER/gdi-pairloss-cnn-20261101','/abr/RESEARCH_USER/gdi-pairloss-cnn-20261101/revision2')
    source=source.replace("studies/20261101-gdi-pairloss-cnn')","studies/20261101-gdi-pairloss-cnn/revision2')")
    source=source.replace("INPUT=H.parent/'20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/data.npz'","INPUT=Path('D:/Projects/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/data.npz')")
    source=source.replace("name='gdi-pairloss-%d'","name='gdi-pairloss-r2-%d'")
    source=source.replace("id='cnn-pairloss-arm%d'","id='cnn-pairloss-r2-arm%d'")
    (R/'manage.py').write_text(source,encoding='utf-8',newline='\n')
    source=(H/'compare.py').read_text(encoding='utf-8').replace("AUDIT=H.parent/'20261101-gdi-longitudinal-audit'","AUDIT=H.parents[1]/'20261101-gdi-longitudinal-audit'")
    (R/'compare.py').write_text(source,encoding='utf-8',newline='\n')
    approval=json.loads((H/'transfer-approval.json').read_text(encoding='utf-8'))
    approval.update(approved=False,source=None,prior_approval_source=approval.get('source'),destination='/abr/RESEARCH_USER/gdi-pairloss-cnn-20261101/revision2',scope='Same public metadata and training input; numerical correction in a child folder; pending explicit approval required by automatic review')
    save(R/'transfer-approval.json',approval)
    print(json.dumps({'revision':str(R),'arms':[0,1],'original_files_modified':False}))
if __name__=='__main__':main()
