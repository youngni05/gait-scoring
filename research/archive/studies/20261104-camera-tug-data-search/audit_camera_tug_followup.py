from pathlib import Path
import sys,json,requests,concurrent.futures,csv,hashlib,io,zipfile,collections,re
sys.stdout.reconfigure(encoding='utf-8')
H=Path('D:/Projects/D_P/studies/20261104-camera-tug-data-search');B=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
sources={
 'mmu_meta':'https://www.kaggle.com/api/v1/datasets/view/limzhekhae1008/mmu-frip',
 'mmu_files':'https://www.kaggle.com/api/v1/datasets/list/limzhekhae1008/mmu-frip',
 'pd_meta':'https://www.kaggle.com/api/v1/datasets/view/insyirahazman/parkinsons-disease-dataset',
 'pd_files':'https://www.kaggle.com/api/v1/datasets/list/insyirahazman/parkinsons-disease-dataset',
 'pd_pose_meta':'https://www.kaggle.com/api/v1/datasets/view/chenhui0331/mmu-parkinsons-disease-tug-test-pose-dataset',
 'pd_pose_files':'https://www.kaggle.com/api/v1/datasets/list/chenhui0331/mmu-parkinsons-disease-tug-test-pose-dataset',
 'wear_access':'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn52540892/wiki/623752',
 'wear_description':'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn52540892/wiki/623753',
 'wear_faq':'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn52540892/wiki/630606',
 'wear_video':'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn52540892/wiki/623758',
 'kinecal71':'https://physionet.org/files/kinecal/1.0.3/kinecal/71/',
 'kinecal71tug':'https://physionet.org/files/kinecal/1.0.3/kinecal/71/71_Get-Up-And-Go-Front-View/',
}
def get(kv):
    k,url=kv
    try:
        with requests.get(url,stream=True,timeout=35) as r:
            body=bytearray()
            for c in r.iter_content(65536):
                body.extend(c)
                if len(body)>3*1024**2:raise ValueError('cap3MiB')
            body=bytes(body);path=B/(k+'.json');path.write_bytes(body);out=dict(key=k,url=url,status=r.status_code,bytes=len(body),sha256=sha(body),path=str(path))
            try:out['data']=json.loads(body)
            except ValueError:out['text']=body.decode('utf8',errors='replace')[:4000]
            return out
    except Exception as e:return dict(key=k,url=url,error=repr(e))
r=list(concurrent.futures.ThreadPoolExecutor(4).map(get,sources.items()));save(H/'followup-http-audit.json',r)
for item in r:
    print(item['key'],item.get('status'),item.get('error',''))
    d=item.get('data',{})
    if 'markdown' in d:print(d['markdown'][:11000])
    elif item['key'].endswith('meta'):print(json.dumps({k:v for k,v in d.items() if k in ['title','subtitle','description','licenseName','licenses','usabilityRating','totalBytes','id','lastUpdated']},ensure_ascii=False)[:14000])
    elif item['key'].endswith('files'):print(json.dumps(d,ensure_ascii=False)[:10000])
    else:print(item.get('text','')[:3000])
d=list(csv.DictReader((B/'new_opencap_csv.txt').read_text(encoding='utf-8-sig').splitlines()));counts={};good=[]
for row in d:
    try:t=float(row['tft_tug']);a=float(row['age']);good.append((row['subid'],t,a))
    except ValueError:pass
counts=dict(rows=len(d),people=len(set(x['subid'] for x in d)),with_tug_rows=len(good),with_tug_people=len(set(x[0] for x in good)),tug65plus_people=len(set(x[0] for x in good if x[2]>=65)),columns=list(d[0]),source_sha256=sha((B/'new_opencap_csv.txt').read_bytes()));save(H/'nmd415-csv-audit.json',counts);print('NMD',counts)
