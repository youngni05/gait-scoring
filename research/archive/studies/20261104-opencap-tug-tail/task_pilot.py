"""Small nine-person additional-task linkage pilot; never used as predictors."""
from pathlib import Path
import json, struct, zlib, hashlib, time, concurrent.futures as cf
import requests, numpy as np
HERE=Path('D:/Projects/D_P/studies/20261104-opencap-tug-tail')
HEAVY=Path('D:/gait-research-data/D_P/opencap-tug-tail-20261104')
INDEX=Path('D:/Projects/D_P/gdi-research-harness/studies/20261030-opencap415-tug-timeseries/archive-index.json')
URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip'
def main():
    diag=json.loads((HERE/'diagnosis.json').read_text());high={r['id']:r['true_TUG'] for r in diag['severe_cases']}
    entries={r['name']:r for r in json.loads(INDEX.read_text())}
    selected=[r for r in diag['additional_task_availability_same_session']['tug_cone']['members'] if r['id'] in high]
    def fetch(item):
        x=entries[item['member']];dst=HEAVY/'task-linkage-pilot'/Path(x['name']).name
        if dst.exists():b=dst.read_bytes()
        else:
            start=x['offset'];end=start+x['compressed']+1023
            for attempt in range(4):
                try:
                    r=requests.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=90)
                    assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                    raw=r.content;assert raw[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',raw,26)
                    assert raw[30:30+nl].decode()==x['name']
                    b=zlib.decompress(raw[30+nl+el:30+nl+el+x['compressed']],-15);break
                except Exception:
                    if attempt==3:raise
                    time.sleep(3*(attempt+1))
            assert len(b)==x['size'] and zlib.crc32(b)==x['crc']
            dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b)
        assert len(b)==x['size'] and zlib.crc32(b)==x['crc']
        lines=b.decode().splitlines();a=np.loadtxt(lines[6:]);names=[v.strip() for v in lines[3].split('\t')[2:] if v.strip()]
        assert a.shape[1]==2+3*len(names);t=a[:,1];pose=a[:,2:].reshape(len(a),len(names),3);p=pose[:,names.index('midHip')]
        duration=float(t[-1]-t[0]);target=high[item['id']]
        shoulder=(pose[:,names.index('LShoulder')]+pose[:,names.index('RShoulder')])/2
        return {'id':item['id'],'member':x['name'],'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'CRC_verified':True,
            'clinical_TUG_seconds':target,'whole_record_seconds':duration,'whole_record_shorter_than_clinical_TUG':duration+.1<target,
            'pelvis_horizontal_spans_m':np.ptp(p[:,[0,2]],axis=0).tolist(),
            'vertical_pelvis_range_m':float(np.ptp(p[:,1])),'shoulder_minus_pelvis_height_range_m':np.ptp(shoulder[:,1]-p[:,1]).item()}
    with cf.ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(fetch,selected))
    result={'n':len(out),'shorter_than_clinical_TUG':sum(r['whole_record_shorter_than_clinical_TUG'] for r in out),'records':out,
        'interpretation':'Archive task name alone does not guarantee that the clinical TUG stopwatch label times this specific recorded repetition. No full-trial duration or score-derived feature is used for training.'}
    (HERE/'additional-task-linkage-pilot.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
