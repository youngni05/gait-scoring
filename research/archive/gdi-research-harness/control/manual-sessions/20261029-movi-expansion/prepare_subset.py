"""Prepare the user-approved ~10 GB MoVi subset on D:, without deleting C: files."""
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('D:/gait-research-data/MoVi')
PILOT=HERE.parent/'20261028-movi-totalcapture-acquisition'
metadata=json.loads((HERE/'movi-metadata-current.json').read_text(encoding='utf-8'))
files=metadata['data']['latestVersion']['files']
pairs={}
for f in files:
    match=re.fullmatch(r'F_PG([12])_Subject_(\d+)_L\.avi',f['label'])
    if match:pairs.setdefault(int(match[2]),{})[int(match[1])]=f
available=sorted(k for k,v in pairs.items() if len(v)==2)
selected=[available[round(i*(len(available)-1)/9)] for i in range(10)]
chosen=[f for subject in selected for f in pairs[subject].values()]
chosen += [f for f in files if f['label'].startswith('F_Subjects_') or f['label'] in ['Camera Parameters.tar','README.pdf']]
entries=[]
for f in chosen:
    d=f['dataFile'];name=f['label']
    assert not f.get('restricted') and d['checksum']['type']=='MD5'
    assert re.fullmatch(r'[A-Za-z0-9_. -]+',name) and not name.endswith(('.', ' '))
    priority=0 if name=='README.pdf' else (1 if name=='Camera Parameters.tar' else 3)
    if name=='F_Subjects_1_45.tar':priority=2
    if name=='F_Subjects_46_90.tar':priority=4
    m=re.fullmatch(r'F_PG([12])_Subject_(\d+)_L\.avi',name)
    entries.append({'id':d['id'],'name':name,'bytes':d['filesize'],'md5':d['checksum']['value'],
                    'url':f"https://dataverse.scholarsportal.info/api/access/datafile/{d['id']}",
                    'priority':priority,'subject':int(m[2]) if m else None,'camera':int(m[1]) if m else None})
entries.sort(key=lambda e:(e['priority'],e['subject'] or 0,e['camera'] or 0))
assert len(set(e['name'].casefold() for e in entries))==len(entries)
total=sum(e['bytes'] for e in entries);assert total<9_000_000_000
assert shutil.disk_usage('D:/').free>total+20_000_000_000
ROOT.mkdir(parents=True,exist_ok=True)
for folder in ['raw','reference','experiments','control']:(ROOT/folder).mkdir(exist_ok=True)
manifest={'dataset':'MoVi','doi':'10.5683/SP2/JRHDRN','selected_subjects':selected,'available_paired_subjects':available,
          'selection':'10 evenly spaced positions in sorted paired-video subject IDs; frozen before model results; not a demographic stratification',
          'raw_total_bytes':total,'total_storage_budget_bytes':10_000_000_000,'files':entries,
          'storage':'Raw, extracted reference and new experiment outputs on D; code and existing pilot report stay on C',
          'reference_archives':'Two complete F archives contain references beyond the 10 video participants; extract only selected subjects',
          'license':'Non-commercial scientific research; no redistribution or commercial training; terms in metadata',
          'deferred':['Remaining MoVi videos, smartphone recordings, IMU and AMASS assets','TotalCapture: provider approval required']}
target=ROOT/'control/subset-manifest.json'
if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==manifest,'Existing manifest differs; stop rather than overwrite'
else:target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
shutil.copy2(HERE/'movi-metadata-current.json',ROOT/'control/provider-metadata.json')
copied=[]
for name,source in [('F_PG1_Subject_1_L.avi',PILOT/'F_PG1_Subject_1_L.avi'),('F_PG2_Subject_1_L.avi',PILOT/'F_PG2_Subject_1_L.avi'),('Camera Parameters.tar',PILOT/'Camera_Parameters.tar')]:
    expected=next(e for e in entries if e['name']==name); dest=ROOT/'raw'/name
    if not dest.exists():shutil.copy2(source,dest)
    assert dest.stat().st_size==expected['bytes'] and hashlib.md5(dest.read_bytes()).hexdigest()==expected['md5']
    copied.append(name)
dest=ROOT/'reference/F_v3d_Subject_1.mat'
if not dest.exists():shutil.copy2(PILOT/'F_v3d_Subject_1.mat',dest)
assert hashlib.sha256(dest.read_bytes()).hexdigest()=='44691cf314ac4007215e0f3eacf22244d2050aba06f8a659b545181398442e04'
print(json.dumps({'root':str(ROOT),'subjects':selected,'files':len(entries),'raw_total_bytes':total,'existing_files_copied_and_verified':copied},indent=2))
