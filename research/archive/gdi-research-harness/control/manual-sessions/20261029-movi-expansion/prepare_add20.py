"""Freeze the approved additional 20 paired-video subjects before inference."""
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('D:/gait-research-data/MoVi')
CONTROL=ROOT/'control'/'add20'
old_path=ROOT/'control'/'subset-manifest.json'
old=json.loads(old_path.read_text(encoding='utf-8'))
meta_path=HERE/'movi-metadata-expansion.json'
metadata=json.loads(meta_path.read_text(encoding='utf-8'))
pairs={}
for f in metadata['data']['latestVersion']['files']:
    m=re.fullmatch(r'F_PG([12])_Subject_(\d+)_L\.avi',f['label'])
    if m and not f.get('restricted'):pairs.setdefault(int(m[2]),{})[int(m[1])]=f
remaining=sorted(s for s,cameras in pairs.items() if len(cameras)==2 and s not in old['selected_subjects'])
selected=[remaining[round(i*(len(remaining)-1)/19)] for i in range(20)]
assert selected==[2,8,12,18,22,27,31,36,40,45,49,54,58,63,67,72,76,81,85,89]
entries=[]
for subject in selected:
    for camera in [1,2]:
        f=pairs[subject][camera];d=f['dataFile']
        assert d['checksum']['type']=='MD5'
        entries.append({'id':d['id'],'name':f['label'],'bytes':d['filesize'],'md5':d['checksum']['value'],
                        'url':f"https://dataverse.scholarsportal.info/api/access/datafile/{d['id']}",
                        'subject':subject,'camera':camera})
total=sum(e['bytes'] for e in entries)
assert len(entries)==40 and total==10_057_058_242
assert not (set(selected)&set(old['selected_subjects']))
assert shutil.disk_usage(ROOT).free>total+22_000_000_000
CONTROL.mkdir(exist_ok=True)
def freeze(path,value):
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==value,'Existing frozen file differs'
    else:path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
manifest={'dataset':'MoVi','batch':'add20','selected_subjects':selected,'previous_subjects':old['selected_subjects'],
          'selection':'20 evenly spaced positions in remaining sorted paired-video IDs; frozen before new inference; not demographic stratification',
          'files':entries,'raw_total_bytes':total,'raw_download_budget_bytes':10_100_000_000,
          'reference_archives':'Reuse the two existing checksum-verified F archives; no reference download',
          'metadata_sha256':hashlib.sha256(meta_path.read_bytes()).hexdigest(),
          'previous_manifest_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest()}
freeze(CONTROL/'download-manifest.json',manifest)
combined={**old,'selected_subjects':sorted(old['selected_subjects']+selected),'files':old['files']+entries,
          'raw_total_bytes':old['raw_total_bytes']+total,'total_storage_budget_bytes':21_000_000_000,
          'selection':{'initial':old['selection'],'add20':manifest['selection']},'batch':'initial10-plus-add20'}
freeze(CONTROL/'experiment-manifest.json',combined)
checkpoint=ROOT/'experiments'/'walking-expansion'/'checkpoints'/'initial10'
if not checkpoint.exists():
    checkpoint.mkdir(parents=True)
    for name in ['plan.json','results.json','patterns.json','verification.json','completion.json','visual-review.json']:
        shutil.copy2(checkpoint.parent.parent/name,checkpoint/name)
    shutil.copy2(HERE.parent/'20261028-movi-totalcapture-acquisition'/'experiment-results.html',checkpoint/'experiment-results.html')
print(json.dumps({'subjects':selected,'files':40,'download_bytes':total,'experiment_total_subjects':30,
                  'download_manifest':str(CONTROL/'download-manifest.json'),'checkpoint':str(checkpoint)},indent=2))
