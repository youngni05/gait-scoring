"""Audit identity joins and finite angle coverage; do not calculate gait scores."""
import collections, hashlib, json, math, re
from pathlib import Path
from pypdf import PdfReader
from openpyxl import load_workbook
HERE=Path(__file__).resolve().parent
DATA=Path('D:/gait-research-data/D_P/dataset-audits/20261102-sideview-reference')
PDF=Path('D:/Projects/D_P/outputs/gait-public-access-20261015/KOA-PD-NM/Demographic Data.pdf')
rows=[]
for page in PdfReader(PDF).pages:
    rows += re.findall(r'(\d{3})_(KOA|PD|NM)\s+(Female|Male)\s+(\d+)\s+(\d\.\d+)',page.extract_text())
assert len(rows)==96
groups=[('KOA','EL',15),('KOA','MD',20),('KOA','SV',15),('PD','ML',6),('PD','MD',7),('PD','SV',3),('NM','NM',30)]
people=[];pos=0
for disease,severity,n in groups:
    for id_,dx,sex,age,height in rows[pos:pos+n]:
        assert dx==disease
        people.append(dict(key=f'{disease}_{severity}_{id_}',disease=disease,severity=severity,id=id_,sex=sex,age=int(age),height=float(height)))
    pos+=n
videos=[]
for item in json.loads((HERE/'koa-rar-inventory.json').read_text())['entries']:
    name=Path(item['filename']).name
    if not name.lower().endswith('.mov'):continue
    m=re.fullmatch(r'(\d+)_(KOA|PD|NM)_(\d+)(?:_(EL|MD|SV|ML))?\.MOV',name,re.I)
    assert m,name
    id_,dx,seq,sev=m.groups();sev=sev or 'NM'
    videos.append(dict(key=f'{dx}_{sev}_{id_:0>3}',sequence=int(seq),filename=item['filename'],size=item['file_size']))
keys={p['key'] for p in people}
assert len(keys)==96 and all(v['key'] in keys for v in videos)
for p in people:
    p['sequences']=sorted(v['sequence'] for v in videos if v['key']==p['key'])
    p['nonempty_sequences']=sorted(v['sequence'] for v in videos if v['key']==p['key'] and v['size']>0)
groupstats=[]
for dx,sev,n in groups:
    pp=[p for p in people if p['disease']==dx and p['severity']==sev]
    groupstats.append(dict(disease=dx,severity=sev,n=len(pp),min_age=min(p['age'] for p in pp),max_age=max(p['age'] for p in pp),age65plus=sum(p['age']>=65 for p in pp),mean_age=sum(p['age'] for p in pp)/len(pp),videos=sum(len(p['sequences']) for p in pp)))
koa=dict(people=people,groups=groupstats,videos=len(videos),unpaired=[p for p in people if p['sequences']!=[1,2]])
koa.update(empty_files=[v['filename'] for v in videos if v['size']==0],nonempty_video_count=sum(v['size']>0 for v in videos),people_with_two_nonempty_sequences=sum(len(p['nonempty_sequences'])==2 for p in people),people_with_any_nonempty_sequence=sum(bool(p['nonempty_sequences']) for p in people))
(HERE/'koa-identity-audit.json').write_text(json.dumps(koa,indent=2),encoding='utf-8')
print('KOA',json.dumps({'groups':groupstats,'unpaired':koa['unpaired']}),flush=True)
results=[]
for cohort,name,expected in [('healthy','MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx',138),('stroke','MAT_normalizedData_PostStrokeAdults_v27-02-23.xlsx',50)]:
    p=DATA/name
    w=load_workbook(p,read_only=True,data_only=True)
    sheets=[s for s in w if s.title!='ReadMe'];assert len(sheets)==expected
    subjects=[]
    for s in sheets:
        iterator=s.iter_rows(values_only=True);header=next(iterator)
        ix=[i for i,k in enumerate(header) if k and 'Angles' in str(k)]
        rr=list(iterator)
        counts={header[i]:sum(isinstance(r[i],(float,int)) and math.isfinite(r[i]) for r in rr) for i in ix}
        subjects.append({'sheet':s.title,'angle_finite_counts':counts,'all_angles_complete':all(v==1001 for v in counts.values())})
    results.append(dict(cohort=cohort,file=name,md5=hashlib.md5(p.read_bytes()).hexdigest(),subjects=subjects,complete=sum(s['all_angles_complete'] for s in subjects),n=len(sheets),headers=list(header)))
    print(cohort,'subjects',len(sheets),'complete angles',results[-1]['complete'],flush=True)
(HERE/'188-xlsx-audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
