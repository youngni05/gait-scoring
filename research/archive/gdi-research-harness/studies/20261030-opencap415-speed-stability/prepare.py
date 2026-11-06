"""Freeze inputs, all repeated splits, and the GPU run bundle before training."""
from common import *
import shutil
import zipfile
from sklearn.model_selection import KFold

def main():
    assert not (HERE / 'preparation.json').exists(), 'Preparation is immutable; do not overwrite a run.'
    src = SOURCE / 'all3s'
    prior = json.loads((SOURCE / 'verification.json').read_text())
    assert prior['passed'] and prior['GPU_models_replayed']
    d = dict(np.load(src / 'input.npz'))
    shutil.copy2(src / 'input.npz', HERE / 'input.npz')
    shutil.copy2(src / 'manifest.csv', HERE / 'manifest.csv')
    protocol = {'source': str(src), 'people': 298, 'seconds': 3,
        'target': 'same-visit measured tft_tug seconds', 'primary_model': 'RF',
        'views': {'mean_speed': 'mean horizontal pelvis speed, one scalar',
                  'speed_summary': 'horizontal pelvis speed mean/std/mean absolute adjacent change, 3 scalars',
                  'pelvis_motion': 'same summaries of horizontal speed and vertical velocity, 6 scalars',
                  'full': 'same summaries of all 19 original channels, 57 scalars'},
        'outer_splits': 'original frozen split plus 4 new shuffled person KFold(5) splits',
        'split_seeds': [20261030,20261031,20261032,20261033,20261034],
        'model_seeds': SEEDS, 'crossed_design': '5 outer partitions x 3 training seeds; no best-seed selection',
        'inner': '3 person folds; same inner folds across views/models/seeds within each outer split',
        'RF': {'trees':128,'leaf_grid':[2,5],'max_features':1.0},
        'Ridge': {'alpha_grid':[1.0,100.0],'deterministic':True},
        'CNN': 'original full 19-channel CNN only, 5x3 repetitions, unchanged architecture/optimizer/early stopping; TCN not repeated',
        'baseline': 'training-only mean and lower median, preserves original convention',
        'normalization': 'training people only, nested tuning inside outer training people',
        'error_analysis': {'primary':'RF full input mean absolute OOF error per person across all 15 runs',
                           'high_error':'top 20% = ceil(298*.2)=60 people, deterministic sample-index tiebreak',
                           'TUG_bins_seconds':[0,10,20,None],
                           'metadata':'same-visit age/sex/diag/ambulation/height/weight, used only after prediction',
                           'quality':'finite inputs, source time intervals, speed variability, angular steps; no pose-confidence or manual quality ground truth'},
        'uncertainty':'paired person bootstrap of across-run mean errors, conditional on fitted models; repeated folds/seeds are correlated, no binomial significance test',
        'scope':'exploratory development-data robustness; no new independent external test or clinical validation',
        'source_input_sha256':sha(src/'input.npz')}
    save('protocol.json', protocol)
    splits = []
    for repeat, seed in enumerate(protocol['split_seeds']):
        if repeat == 0:
            items = json.loads((src / 'splits.json').read_text())
        else:
            items = []
            for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(d['ids'])):
                people = np.sort(d['ids'][tr]); inner=[]
                for a,b in KFold(3,shuffle=True,random_state=seed+1).split(people):
                    it=tr[np.isin(d['ids'][tr],people[a])];iv=tr[np.isin(d['ids'][tr],people[b])]
                    inner.append({'train':it.tolist(),'validation':iv.tolist()})
                items.append({'fold':fold,'train':tr.tolist(),'test':te.tolist(),'inner':inner})
        splits.append({'repeat':repeat,'split_seed':seed,'splits':items})
    save('splits.json', splits)
    common_code=(src/'common.py').read_text().replace('ROOT = HERE.parents[3]', 'ROOT = HERE.parents[4]')
    gpu_code=(src/'gpu.py').read_text().replace("['CNN','TCN']", "['CNN']").replace("'models_finished':2", "'models_finished':1")
    gpu_code=gpu_code.replace('torch.set_num_threads(2)', "MODEL_SEED=json.loads((HERE/'run-config.json').read_text())['model_seed']\ntorch.set_num_threads(2)")
    gpu_code=gpu_code.replace('20261030', 'MODEL_SEED')
    gpu_code=gpu_code.replace("(HERE/'pilot-summary.json').read_text()", "(HERE.parents[1]/'pilot-summary.json').read_text()")
    names=[]
    for repeat in splits:
        for seed in SEEDS:
            name=f"r{repeat['repeat']:02d}-s{seed}";names.append(name)
            out=HERE/'cnn'/name;out.mkdir(parents=True)
            dd={k:v.copy() for k,v in d.items()}
            for f in repeat['splits']:dd['folds'][f['test']]=f['fold']
            np.savez_compressed(out/'input.npz', **dd)
            save(out/'splits.json', repeat['splits'])
            save(out/'protocol.json', {**protocol, 'repeat':repeat['repeat'],'model_seed':seed})
            save(out/'run-config.json', {'repeat':repeat['repeat'],'model_seed':seed})
            save(out/'input-audit.json', {'input_sha256':sha(out/'input.npz'),'protocol_sha256':sha(out/'protocol.json')})
            (out/'common.py').write_bytes(common_code.encode())
            (out/'gpu.py').write_bytes(gpu_code.encode())
    save('cnn-runs.json', names)
    slurm='''#!/bin/bash
#SBATCH --job-name=tug-speed-stability
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --output=logs/stability-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [ "$1" = pilot ]; then
  /usr/bin/python3.8 -u cnn/r00-s20261030/gpu.py pilot
  cp cnn/r00-s20261030/pilot-summary.json pilot-summary.json
else
  for run in '''+' '.join(names)+'''; do
    echo "RUN_STARTED $run"
    /usr/bin/python3.8 -u "cnn/$run/gpu.py" train
  done
fi
echo "STABILITY_${1}_FINISHED"
'''
    (HERE/'submit.slurm').write_bytes(slurm.encode())
    save('preparation.json',{'source_input_sha256':sha(src/'input.npz'),'protocol_sha256':sha(HERE/'protocol.json'),
                             'splits_sha256':sha(HERE/'splits.json'),'source_prior_verification':prior})
    files=[p for p in (HERE/'cnn').rglob('*') if p.is_file()]+[HERE/'submit.slurm',HERE/'cnn-runs.json']
    save('bundle-hashes.json',{p.relative_to(HERE).as_posix():sha(p) for p in files})
    with zipfile.ZipFile(HERE/'gpu-bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[HERE/'bundle-hashes.json']:z.write(p,p.relative_to(HERE).as_posix())
    print(json.dumps({'prepared':True,'people':len(d['ids']),'splits':len(splits),'cnn_runs':len(names),'bundle_sha256':sha(HERE/'gpu-bundle.zip')}))

if __name__=='__main__':main()
