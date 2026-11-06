from common import *
import shutil

def main():
    assert not (HERE/'preparation.json').exists()
    previous=json.loads((PREV/'analysis-verification.json').read_text());assert previous['passed']
    for name in ['input.npz','splits.json']:shutil.copy2(PREV/name,HERE/name)
    selection=[s for s in json.loads((PREV/'cpu-selections.json').read_text()) if s['model']=='RF' and s['view']=='speed_summary']
    assert len(selection)==75
    save('baseline-selections.json',selection)
    protocol={'people':298,'seconds':3,'input_sha256':sha(PREV/'input.npz'),'splits_sha256':sha(PREV/'splits.json'),
      'partition_count':5,'model_seeds':SEEDS,'outer_folds':5,'inner_folds':3,
      'baseline':'previous speed-summary RF, exact same inputs/splits/seeds/hyperparameter selection; reused verified predictions and inner OOF predictions',
      'methods':{name:{'view':v,'long_TUG_training_weight':w,'inner_selection':objective,'model':model} for name,(v,w,objective,model) in METHODS.items()},
      'RF':{'n_estimators':128,'leaf_grid':[2,5],'max_features':1.0,'sample_weight':'explicit ones, or 10 for TRAINING targets >=20; uniform scaler weights for all methods'},
      'Ridge':{'alpha_grid':[.1,1.,10.,100.],'predictions':'nonnegative, no upper cap','deterministic':'5 partitions; not refitted for duplicate training seeds'},
      'features':{'raw':'mean/std/mean absolute adjacent change of horizontal pelvis speed',
                  'robust':'median/IQR/median absolute adjacent change of same speed window, no target-dependent filtering',
                  'inverse':'1/max(mean_speed,.1), speed_std/max(mean_speed,.1), mean_speed_change/max(mean_speed,.1)'},
      'selection':'pooled INNER held-out predictions on outer training people; MAE or equal mean of errors in <10,10-<20,>=20 second bands present in training. No outer labels in fitting/selection.',
      'analysis':'long>=20, mid10-<20, short<10; MAE/RMSE/signed error/underprediction/overprediction for every method; all repetitions reported, no best seed selection',
      'cause_audit':'clinical join, translation spikes, train tail counts/range, feature-nearest training neighbors, exact baseline forest training-label weights and tail influence',
      'scope':'exploratory reuse of already examined development cohort; nine unique long-TUG people, repeated predictions are correlated; not new clinical or external validation',
      'device':'local CPU only; no external transfer','new_training_inputs':'none: no diagnosis, age, other timed clinical tests, target band at inference or target-dependent window exclusions'}
    save('protocol.json',protocol)
    save('preparation.json',{'hashes':{n:sha(HERE/n) for n in ['input.npz','splits.json','protocol.json','baseline-selections.json']},'previous_validation':previous})
    d=load();assert np.array_equal(d['x'],np.load(PREV/'input.npz')['x'])
    print('FROZEN_TAIL_PROTOCOL',len(d['y']),int((d['y']>=20).sum()))

if __name__=='__main__':main()
