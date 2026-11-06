from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import pandas as pd
SRC=ROOT/'gdi-research-harness/studies/20261031-mobilised-uncertainty-review'
a=pd.read_csv(SRC/'extracted/Main datasets for analysis/CSV files/All_cohorts_dataset.csv',low_memory=False)
print('DMO columns', [c for c in a if c.endswith('_w')])
for c in ['1crtsafe','1crtresult','5crtsafe','5crt_score']:
 print(c,a[c].value_counts(dropna=False).to_dict())
print('positive times by safe and single',a[a['5crtresult']>0].groupby(['5crtsafe','1crtresult'],dropna=False).size().to_dict())
print('time score mismatches')
t=a['5crtresult'];expected=pd.Series(0,index=a.index)
expected.loc[t<=60]=1;expected.loc[t<16.7]=2;expected.loc[t<13.7]=3;expected.loc[t<11.2]=4
m=(t>0)&(expected!=a['5crt_score'])
print(a.loc[m,['participantid','visit.number','5crtsafe','1crtresult','5crtresult','5crt_score']].to_string(index=False))
print('small times',a.loc[(t>0)&(t<3),['participantid','visit.number','5crtsafe','1crtresult','5crtresult','5crt_score']].to_dict('records'))
print('selected existing studies protocol')
p=ROOT/'gdi-research-harness/studies/20261031-mobilise-components/protocol.json'
if p.exists():print(p.read_text(encoding='utf8')[:2000])
