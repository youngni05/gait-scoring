"""Freeze the already approved people/features and a separate larger sensor cohort."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path('D:/Projects/D_P/studies/20261103-sppb-tabular-foundation')
BIG=Path('D:/gait-research-data/D_P/sppb-tabular-foundation-20261103')
BASE=Path('D:/Projects/D_P/studies/20261103-sppb-lightgbm')
SMALL=Path('D:/gait-research-data/D_P/sppb-lightgbm-20261103/cohorts.json')
SRC=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-mobilised-uncertainty-review/extracted')
ELIG=Path('D:/Projects/D_P/gdi-research-harness/studies/20261031-mobilised-uncertainty-review/first-eligible-per-target.csv')
SPEED=['ws_30_p90_w','ws_1030_avg_w','ws_10_p90_w','ws_30_avg_w','ws_30_var_w']
PATTERN=['strlen_1030_avg_w','strlen_30_avg_w','strlen_30_var_w','cadence_all_avg_w','cadence_all_var_w','cadence_30_avg_w','cadence_30_p90_w','strdur_all_avg_w','strdur_all_var_w','strdur_30_avg_w']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    BIG.mkdir(exist_ok=True,parents=True)
    cohorts=read(SMALL)
    assert sha(SMALL)=='ae85dbea1968ef75f8b982217be998498e7fd42ce4f8ef5dc83dc565fd4279a7'
    main=SRC/'Main datasets for analysis/CSV files/All_cohorts_dataset.csv'
    a=pd.read_csv(main,low_memory=False);e=pd.read_csv(ELIG);e=e[e.audit_target=='SPPB']
    keys=['participantid','visit.number']; assert not a.duplicated(keys).any()
    # Reproduce the original eligibility and first-visit selection independently.
    eligible=(a.ws_1030_avg_w>0)&(a.n_days_w>=3)&a.proto_dev.eq('N')&a.total_sppb_score.between(0,12)
    x=a[eligible].sort_values(keys).drop_duplicates('participantid')
    assert set(map(tuple,x[keys].to_numpy()))==set(map(tuple,e[keys].to_numpy()))
    assert len(x)==2232 and x.participantid.nunique()==2232
    components=x[['balance_score','4mwt_speed_score','5crt_score']]
    complete=components.notna().all(axis=1)
    np.testing.assert_allclose(components[complete].sum(axis=1),x.loc[complete,'total_sppb_score'])
    # Official totals remain the target under the original eligibility rule.
    # Do not invent missing component scores or silently call them verified.
    incomplete=x.loc[~complete,keys+list(components)+['total_sppb_score']].replace({np.nan:None}).to_dict('records')
    # Same source visit for features, age and target. Age/disease used only for reporting.
    x=x.assign(id=x.participantid.astype(str)).sort_values('id').reset_index(drop=True)
    fold=np.full(len(x),-1)
    for f,(_,te) in enumerate(KFold(5,shuffle=True,random_state=20261103).split(x)):fold[te]=f
    people=[]
    for i,row in x.iterrows():
        features={f:float(row[f]) if pd.notna(row[f]) else None for f in SPEED+PATTERN}
        people.append(dict(id=row.id,dataset='mobilised',sppb=float(row.total_sppb_score),age=float(row.age),cohort=row.cohort,visit=row['visit.number'],fold=int(fold[i]),features=features,gdi=None))
    cohorts['mobilised']=people;save(BIG/'cohorts.json',cohorts)
    audit=dict(n=2232,age65plus=int((x.age>=65).sum()),age_range=[float(x.age.min()),float(x.age.max())],disease_counts=x.cohort.value_counts().to_dict(),sppb_counts={str(k):int(v) for k,v in x.total_sppb_score.value_counts().sort_index().items()},feature_missing={f:int(x[f].isna().sum()) for f in SPEED+PATTERN},original_components_sum_checked=int(complete.sum()),official_total_with_incomplete_components=incomplete,first_eligible_reproduced=True)
    save(HERE/'inventory.json',audit)
    baseline_protocol=read(BASE/'protocol.json')
    protocol=dict(owner_thread='01a19c6f-49f7-7300-97b0-a303b71ffe91',frozen_before_predictions=True,
      purpose='Compare fixed TabICLv2 and TabPFN v2 with LightGBM on identical people, features and folds; separate large weekly-sensor extension from short-walk datasets.',
      sources={str(p):sha(p) for p in [SMALL,BASE/'comparison.json',BASE/'verification.json',BASE/'protocol.json',main,ELIG,HERE/'weights-manifest.json']},
      cohort_sha256=sha(BIG/'cohorts.json'),
      small='Exact previously frozen older37/older24/older44/gait2care92/gstride158 input rows and outer folds. Existing LightGBM predictions reused, not retuned. These cohorts overlap within OlderAdults and must not be pooled.',
      large='2232 first eligible SPPB visits; >=3 provider-valid sensor days, positive ws_1030_avg_w, proto_dev=N, valid official SPPB 0..12. One row/person. Sorted string IDs KFold5 shuffle seed20261103. Four disease cohorts, 1222 aged>=65; neither healthy-community nor short-camera validation. Same folds for all models. CC BY-NC-ND4.0 source, internal research only; no product-use permission inferred.',
      conditions=dict(small=['speed','features','features_GDI only older37/older24'],mobilised=dict(speed=[SPEED[0]],speed_profile=SPEED,features=SPEED+PATTERN)),
      model=dict(TabICLv2=dict(checkpoint='tabicl-regressor-v2-20260212.ckpt',n_estimators=8,device='cpu',random_state=20261103),TabPFNv2=dict(checkpoint='tabpfn-v2-regressor-v2_default.ckpt',version='V2',n_estimators=8,device='cpu',random_state=20261103),notes='Fixed pretrained weights; no gradient training or outcome-driven hyperparameter/ensemble tuning. Native training-only preprocessing and missing-value handling. CPU float32 where supported. Mean prediction; no SPPB clipping/calibration. Model inputs never include age, diagnosis, site, SPPB subscores or test walking time. CPU sample-count runtime guard may be disabled for 1786 training people, within v2 pretraining limits; runtime pilot before full large run.'),
      LGBM=baseline_protocol['LGBM'],
      LGBM_large='Same 16-setting grid as prior baseline, inner KFold3 seed20261031, choose pooled inner MAE, first tie. All tuning inside outer training. Add training mean baseline.',
      metrics='Person MAE primary; RMSE,bias,R2,Spearman,within1,errors>2,bands0-4/5-8/9-10/11-12 and <=8 confusion. Paired bootstrap10000 seed20261103 conditional on fixed OOF predictions. Within-model feature additions and matched-feature model differences. Report >=65 and diagnosis strata descriptively; no best-subgroup selection.',
      verification='Check hashes and person/fold/feature identity; save every final fitted state or complete deterministic training context plus frozen checkpoint. Reload/reconstruct and predict every held-out person again, compare predictions; independent metrics/paired intervals and inner selection audit. Preserve original results and all failures.',
      limitations='Reused development cohorts; no external clinical validation. SPPB contains walking component; correlation does not show all physical function is measured. GDI input is prior virtual-camera simulation with anatomical uncertainties, not observed RGB. Sensor cohorts cannot supply GDI.',
      storage=dict(small=str(HERE),large=str(BIG),expected_GB=3))
    assert not (HERE/'protocol.json').exists(),'Do not overwrite frozen protocol'
    save(HERE/'protocol.json',protocol)
    save(HERE/'preparation-receipt.json',dict(protocol_sha256=sha(HERE/'protocol.json'),cohorts_sha256=sha(BIG/'cohorts.json'),code_sha256=sha(Path(__file__))))
    print(json.dumps(audit,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
