"""Independently verify within-score, within-fold, distinct-person rankings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
labels={'balance':'balance_score','chair':'5crt_score','remaining':'remaining'}
columns=['band','k','seed','draw','fold','participantid','4mwt_speed_score','balance_score','5crt_score','RF_balance','RF_chair','DS_balance','DS_chair']
p=pd.read_csv(HERE/'predictions.csv.gz',usecols=columns)
p=p[p.draw.eq(0)].copy()
p['remaining']=p[['balance_score','5crt_score']].sum(axis=1,min_count=2)
p['RF_remaining']=p.RF_balance+p.RF_chair
p['DS_remaining']=p.DS_balance+p.DS_chair
results=pd.read_csv(HERE/'within-gait-ranking.csv').set_index(['band','k','seed','gait_score','endpoint','model'])
count=0
for (band,k,seed,gait),d in p.groupby(['band','k','seed','4mwt_speed_score']):
    for endpoint,actual in labels.items():
        columns=[actual,'RF_'+endpoint,'DS_'+endpoint]
        people=d.dropna(subset=columns).groupby(['fold','participantid'])[columns].mean().reset_index()
        wins={'RF':0.,'DS':0.};pairs=0
        for _,g in people.groupby('fold'):
            # Partition by true person-average score, then compare every lower
            # score group with strictly higher groups. This avoids the analysis
            # implementation's upper-triangle difference-product calculation.
            for value,lower in g.groupby(actual):
                higher=g[g[actual]>value+1e-12]
                pairs+=len(lower)*len(higher)
                for model in ['RF','DS']:
                    lo=lower[model+'_'+endpoint].to_numpy()[:,None]
                    hi=higher[model+'_'+endpoint].to_numpy()[None,:]
                    wins[model]+=float((hi>lo).sum()+.5*(hi==lo).sum())
        for model in ['RF','DS']:
            row=results.loc[(band,k,seed,int(gait),endpoint,model)]
            assert row.people==len(people) and row.pairs==pairs
            if pairs:np.testing.assert_allclose(wins[model]/pairs,row.concordance,atol=1e-12,rtol=1e-12)
            else:assert pd.isna(row.concordance)
            count+=1
assert count==len(results)==900
record={'status':'passed','ranking_rows_independently_recomputed':count,'unit':'one average per person within the observed gait-score group; compare different people only within the same outer fold','constant_lookup_concordance':.5,'ties_in_true_scores':'excluded','ties_in_predictions':'half credit'}
(HERE/'ranking-verification.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record),flush=True)
