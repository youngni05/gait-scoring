"""Prepare an isolated matched MSE/conv-transfer bundle from audited existing code."""
from pathlib import Path
import os,json,hashlib,ast
import numpy as np
ROOT=Path('D:/Projects/D_P');HERE=ROOT/'gdi-research-harness/studies/20261101-synthgait-conv-transfer'
OLD=ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-person-bmc692'
SRC=ROOT/'gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20'
DATA=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261101-synthgait-conv-transfer')
VIZ=Path('C:/Users/jun/.codex/visualizations/2026/10/30/01a18b12-4a31-7283-a35d-17f28017111a')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def replace_once(text,a,b):
    assert text.count(a)==1,(a,text.count(a));return text.replace(a,b)
def main():
    HERE.mkdir(exist_ok=False);DATA.mkdir(parents=True,exist_ok=True)
    report=json.loads((SRC/'feature-verification.selected.json').read_text(encoding='utf-8'))
    final=json.loads((SRC/'final-verification.json').read_text(encoding='utf-8'));assert final['artifact_audit_passed']
    rows=[r for r in report['records'] if r['max_required_joint_missing_fraction']<=.25]
    arrays={};sources=[]
    for i,r in enumerate(rows):
        p=Path(r['features_path']);assert sha(p)==r['features_sha256']
        with np.load(p,allow_pickle=False) as z:
            arrays['x%02d'%i]=z['x_native'];assert np.isfinite(arrays['x%02d'%i]).all()
            sources.append(dict(sample_id=r['sample_id'],walk_id=r['walk_id'],split=r['split'],cadence=float(z['cadence']),
                                missing_fraction=r['max_required_joint_missing_fraction'],path=str(p),sha256=sha(p)))
    arrays.update(walk_id=np.array([r['walk_id'] for r in sources]),split=np.array([r['split'] for r in sources]),
                  cadence=np.array([r['cadence'] for r in sources]),missing_fraction=np.array([r['missing_fraction'] for r in sources]),fps=np.full(len(rows),16.))
    np.savez_compressed(DATA/'external.npz',**arrays)
    assert len(rows)==18 and sum(r['split']=='train' for r in sources)==14
    save(HERE/'external-provenance.json',{'source_report_sha256':sha(SRC/'feature-verification.selected.json'),
        'source_final_verification_sha256':sha(SRC/'final-verification.json'),'records':sources,
        'excluded_quality_records':[r['sample_id'] for r in report['records'] if r not in rows],
        'external_npz':str(DATA/'external.npz'),'external_sha256':sha(DATA/'external.npz'),
        'no_resampling':True,'no_GDI_label':True,'no_heel_strike_loss':True,'subject_disjointness_not_verifiable':True})
    for name in ['model.py','windows.py','balanced.py','audit.py','input_contract.json']:(HERE/name).write_bytes((OLD/name).read_bytes())
    code=(OLD/'run.py').read_text(encoding='utf-8')
    code=replace_once(code,"require(c['stage'] == 'person_balanced_mse_bmc_fixed_sigma', 'Baseline scope changed')","require(c['stage'] == 'synth_conv_transfer', 'Transfer scope changed')")
    code=replace_once(code,"    model = GdiCNN(c['dropout']).to(device)\n    opt = make_optimizer(model, c)",
        "    model = GdiCNN(c['dropout']).to(device)\n    import stem\n    proof = stem.initialize(model, c, identity['fit_seed'], ROOT)\n    proof_path = folder / 'initialization.json'\n    if proof_path.exists():\n        require(load(proof_path) == proof, 'Initialization provenance changed')\n    else:\n        save(proof_path, proof)\n    opt = make_optimizer(model, c)")
    code=replace_once(code,"    return {'bundle_sha256': sha(ROOT / 'manifest.json'), 'purpose': purpose,",
        "    import stem\n    return {'initialization':c['initialization'], 'pretrain_checkpoint_sha256':stem.source_sha(ROOT,c,seed),\n            'bundle_sha256': sha(ROOT / 'manifest.json'), 'purpose': purpose,")
    code=replace_once(code,"choices=['mse','bmc'],required=True","choices=['scratch','synth'],required=True")
    code=replace_once(code,"    c = config();c.update(loss_variant=args.variant,name='cnn_'+args.variant+'_person_balanced')",
        "    c = config_for(args.variant, 'pilot' if args.command == 'pilot' else 'main')")
    marker='\ndef person_mae(y, pred, people):'
    code=replace_once(code,marker,"\ndef config_for(variant, stage='main'):\n    require(variant in ['scratch','synth'] and stage in ['pilot','main'],'Invalid transfer arm/stage')\n    c=config();c.update(loss_variant='mse',initialization=variant,pretraining_stage=stage,name='cnn_'+variant+'_conv_transfer')\n    return c\n\n"+marker)
    (HERE/'run.py').write_text(code,encoding='utf-8')
    audit=(HERE/'audit.py').read_text(encoding='utf-8');audit=replace_once(audit,"    done=r.load(folder/'fit_complete.json');h=r.load(folder/'curve.json')",
        "    proof=r.load(folder/'initialization.json')\n    r.require(proof['initialization']==c['initialization'] and proof['all_other_state_unchanged'] and proof['batchnorm_reset'] and proof['rng_unchanged'],'Wrong transfer provenance')\n    if c['initialization']=='synth':\n        r.require(proof['pretrain_checkpoint_sha256']==ident['pretrain_checkpoint_sha256'],'Changed pretrained convolution checkpoint')\n    done=r.load(folder/'fit_complete.json');h=r.load(folder/'curve.json')")
    (HERE/'audit.py').write_text(audit,encoding='utf-8')
    review=(OLD/'review.py').read_text(encoding='utf-8').replace("choices=['mse','bmc']","choices=['scratch','synth']")
    review=replace_once(review,"c=r.config();c.update(loss_variant=a.variant,name='cnn_'+a.variant+'_person_balanced')","c=r.config_for(a.variant,'pilot')")
    (HERE/'review.py').write_text(review,encoding='utf-8')
    for src,name in [('synth_conv_stem.py','stem.py'),('synth_conv_pretrain_cli.py','pretrain_cli.py'),('synth_conv_selfcheck.py','selfcheck.py'),('synth_conv_worker.py','worker.py')]:
        (HERE/name).write_bytes((VIZ/src).read_bytes())
    c=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    c.update(name='synthgait_native_conv_transfer',stage='synth_conv_transfer',variants=['scratch','synth'],workers=1,
        external_data_sha256=sha(DATA/'external.npz'),pretrain_max_epochs=30,
        pretrain_label='clip cadence only',pretrain_train_walks=14,pretrain_validation_walks=4,
        loss='Same person-balanced MSE in both GDI arms',expected_external_fits=25,expected_fits=50,expected_outer_folds=10,
        note='Matched seed per fit before external training; transfer only six convolution weights/biases. All BN and dense states reset. Fixed native16fps; GDI original124frames unchanged. External cohort overlap cannot be checked by subject IDs.')
    save(HERE/'protocol.json',c)
    (HERE/'submit.slurm').write_text('''#!/usr/bin/env bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --exclude=DIS04,DIS05,DIS06
#SBATCH --job-name=gdi-synth-conv
#SBATCH --output=logs/worker-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /usr/bin/env /usr/bin/python3.8 -u -B worker.py
''',encoding='utf-8',newline='\n')
    save(HERE/'source-provenance.json',{'base':str(OLD),'unchanged_files':{n:sha(OLD/n) for n in ['model.py','windows.py','balanced.py','input_contract.json']},
        'base_runner_sha256':sha(OLD/'run.py'),'base_audit_sha256':sha(OLD/'audit.py'),'base_review_sha256':sha(OLD/'review.py'),
        'changes':['Only add initialization proof and convolution transplant to GDI runner','Stage/arm names','Pretrain checkpoint identity and audit','New auxiliary short-input model/worker']})
    (HERE/'build.py').write_bytes(Path(__file__).read_bytes())
    (HERE/'SYNTHGAIT_LICENSE').write_bytes((SRC.parent/'metadata/LICENSE').read_bytes())
    (HERE/'CNN_LICENSE').write_bytes((OLD/'sources/LICENSE').read_bytes())
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    names=[p.name for p in HERE.iterdir() if p.suffix in ['.py','.json','.slurm'] and p.name!='manifest.json']
    save(HERE/'manifest.json',{n:sha(HERE/n) for n in sorted(names)})
    print(json.dumps({'folder':str(HERE),'data':str(DATA/'external.npz'),'external_train_walks':14,'validation_walks':4,'manifest_sha256':sha(HERE/'manifest.json')}))
if __name__=='__main__':main()
