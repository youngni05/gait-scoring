"""Independent arithmetic and artifact audit of the completed comparison."""
import json,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'feature-analysis/deps'))
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((HERE/'analysis.json').read_text());z=np.load(HERE/'scores.npz')
assert d['passed'] and sha(HERE/'scores.npz')==d['prediction_sha256']
assert '33003|COMPLETED|0:0|' in (HERE/'slurm-final.txt').read_text()
receipt=json.loads((HERE/'collection-verification.json').read_text());assert receipt['passed'] and sha(HERE/'review.tar.gz')==receipt['sha256']
pre=json.loads((HERE/'preprocessing-verification.json').read_text());assert pre['passed'] and pre['adapter_sha256']==sha(HERE/'analyze_video.py')
for p,h in d['model_sha256'].items():assert sha(HERE.parent/p)==h
assert len(d['model_sha256'])==60
keys=list(z['keys']);ids={k:i for i,k in enumerate(keys)}
protocol=json.loads((HERE/'protocol.json').read_text());subjects=[v['subject'] for v in protocol['selected']]
clips=json.loads((HERE/'clips.json').read_text());checks=[];byseed={}
for name,array,summary in [('motion128',z['predictions'],d['summary']),('original698',z['original_predictions'],d['original_698_video_summary'])]:
    assert array.shape==(30,70) and np.isfinite(array).all()
    paired=np.array([array[:,ids['s%02d_PG2_clean'%s]]-array[:,ids['s%02d_PG1_clean'%s]] for s in subjects])
    error=abs(np.abs(paired).mean()-summary['view_abs_difference']['mean']);assert error<=1e-12;checks.append(float(error))
    byseed[name]=dict(camera=[float(np.abs(paired[:,np.array(z['seed'])==seed]).mean()) for seed in sorted(set(z['seed']))],PG2_higher_count=int((paired>0).sum()),view_comparisons=int(paired.size),jitter={})
    for sigma in [.02,.05]:
        change=np.array([array[:,ids[c['key']+'_'+v['name']]]-array[:,ids[c['key']+'_clean']] for c in clips for v in protocol['conditions'] if v['sigma']==sigma])
        error=abs(abs(change).mean()-summary['jitter'][str(sigma)]['absolute_change']['mean']);assert error<=1e-12;checks.append(float(error))
        byseed[name]['jitter'][str(sigma)]=[float(abs(change[:,np.array(z['seed'])==seed]).mean()) for seed in sorted(set(z['seed']))]
clean=[ids[c['key']+'_clean'] for c in clips];p=z['predictions'][:,clean]
result=dict(passed=True,people=5,clips=10,conditions=70,frames=10500,primary_models=30,reference_models=30,max_summary_error=max(checks),RF_replay_error=d['RF_replay_error'],feature_reproduction_error=d['feature_reproduction_error'],source_platform_noise_replay_error=d['source_platform_noise_replay_error'],cross_platform_noise_max_difference=d['cross_platform_noise_max_difference'],all_clean_rounded_to_one=bool((np.floor(p+.5)==1).all()),clean_clip_model_mean_range=[float(p.mean(0).min()),float(p.mean(0).max())],by_seed=byseed,analysis_sha256=sha(HERE/'analysis.json'),scores_sha256=sha(HERE/'scores.npz'))
(HERE/'final-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
