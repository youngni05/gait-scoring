"""Download only age-linked TD sagittal gait videos; validate each ZIP CRC."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import struct
import zlib
import openpyxl
from inspect_public_video import fetch

ROOT=Path(__file__).resolve().parent
def download(row):
    name=row['name'];dest=ROOT/'td_sagittal'/Path(name).name
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        payload=dest.read_bytes()
        assert len(payload)==row['size'] and zlib.crc32(payload)==row['crc']
    else:
        start=row['header_offset']
        # Variable local header plus compressed payload; a small suffix is harmless.
        blob,cr=fetch(str(start)+'-'+str(start+row['compressed_size']+4095))
        assert int(cr.split()[1].split('-')[0])==start
        header=struct.unpack('<4s5H3L2H',blob[:30]);assert header[0]==b'PK\x03\x04'
        n,e=header[-2:];decoded=blob[30:30+n].decode('utf-8')
        assert decoded==name
        raw=blob[30+n+e:30+n+e+row['compressed_size']]
        assert row['compression']==8
        payload=zlib.decompress(raw,-15)
        assert len(payload)==row['size'] and zlib.crc32(payload)==row['crc']
        dest.write_bytes(payload)
    return dict(file=dest.name,person=name.split('/')[2],size=len(payload),sha256=hashlib.sha256(payload).hexdigest(),zip_crc_verified=True)
def main():
    wb=openpyxl.load_workbook(ROOT/'PARTICIPANT_CHARACTERISTICS.xlsx',read_only=True,data_only=True)
    ages={r[1]:float(r[4]) for r in list(wb['TD'].values)[1:] if r[1] and isinstance(r[4],(int,float))}
    inventory=json.loads((ROOT/'video_inventory.json').read_text())
    candidates=[r for r in inventory['files'] if r['name'].startswith('VIDEO/TD/') and 'GAIT' in r['name'] and 'SAGITTAL.MOV' in r['name']]
    video_people={r['name'].split('/')[2] for r in candidates}
    selected=[r for r in candidates if r['name'].split('/')[2] in ages]
    result={'metadata_without_video':sorted(set(ages)-video_people),'video_without_age':sorted(video_people-set(ages)),
            'eligible_people':len(set(ages)&video_people),'expected_videos':len(selected),'videos':[]}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(download,r):r for r in selected}
        for f in as_completed(futures):
            row=f.result();row['age_years']=ages[row['person']];result['videos'].append(row)
            (ROOT/'td_download.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps({'downloaded':len(result['videos']),'expected':len(selected),'file':row['file']}),flush=True)
    assert len(result['videos'])==len(selected)
    print(json.dumps({'complete':True,'people':result['eligible_people'],'videos':len(selected)}))
if __name__=='__main__':main()
