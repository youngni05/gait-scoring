"""Small public metadata/file audit. Never download complete video archives."""
from pathlib import Path
import requests,json,hashlib,time,concurrent.futures,csv,io,zipfile,collections
H=Path('D:/Projects/D_P/studies/20261104-camera-tug-data-search');B=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104')
H.mkdir(exist_ok=True);B.mkdir(exist_ok=True)
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
def sha(b):return hashlib.sha256(b).hexdigest()
URLS={
 'kaggle_mmu_list':'https://www.kaggle.com/api/v1/datasets/list?search=mmu%20fall',
 'kaggle_mmu_files':'https://www.kaggle.com/api/v1/datasets/list/limzhekhae1008/mmu-fall-riREMOVED_SECRET',
 'kaggle_timed_list':'https://www.kaggle.com/api/v1/datasets/list?search=timed%20up%20go',
 'zenodo415':'https://zenodo.org/api/records/22309771',
 'zenodo129':'https://zenodo.org/api/records/13788592',
 'synapse_wiki':'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn52540892/wiki/623751',
 'synapse_headers':'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn52540892/wikiheadertree',
 'kinecal_register':'https://physionet.org/files/kinecal/1.0.3/register.csv',
 'nmd_readme':'https://raw.githubusercontent.com/stanfordnmbl/NMD_Detection_Quantification/main/README.md',
 'old_opencap_readme':'https://zenodo.org/records/13788592/files/README.md?download=1',
 'new_opencap_csv':'https://zenodo.org/records/22309771/files/nmd_opencap_participant_info.csv?download=1',
 'mmu_article':'https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1425713/full',
}
def fetch(item):
    key,url=item
    try:
        with requests.get(url,stream=True,timeout=35,headers={'User-Agent':'Research metadata audit'}) as r:
            chunks=[];n=0
            for c in r.iter_content(65536):
                n+=len(c)
                if n>2*1024**2:raise ValueError('metadata cap2MiB; no complete archive downloaded')
                chunks.append(c)
            body=b''.join(chunks);path=B/(key+'.txt');path.write_bytes(body);result=dict(key=key,url=url,status=r.status_code,content_type=r.headers.get('Content-Type'),bytes=len(body),sha256=sha(body),saved=str(path))
            try:
                data=json.loads(body)
                if key.startswith('kaggle'):
                    result['summary']=[dict(ref=x.get('ref'),title=x.get('title'),size=x.get('totalBytes')) for x in data] if isinstance(data,list) else data
                elif key.startswith('zenodo'):result['summary']=dict(title=data.get('metadata',{}).get('title'),license=data.get('metadata',{}).get('license'),files=[dict(key=f['key'],size=f['size'],checksum=f['checksum'],links=f['links']) for f in data.get('files',[])])
                else:result['summary']=data
            except (ValueError,KeyError):pass
            return result
    except Exception as e:return dict(key=key,url=url,error=repr(e))
results=list(concurrent.futures.ThreadPoolExecutor(4).map(fetch,URLS.items()));save(H/'http-audit.json',dict(utc=time.time(),scope='Public small metadata/README/CSV only. No patient video archive downloaded; no auth keys used; no contact made.',results=results))
for r in results:
    print(r['key'],r.get('status'),r.get('bytes'),r.get('error',''),flush=True)
    if r['key'].startswith('kaggle') and 'summary' in r:print(str(r['summary'])[:1800],flush=True)
    if r['key'].startswith('zenodo') and 'summary' in r:print(r['summary'],flush=True)
    if r['key']=='new_opencap_csv' and r.get('status')==200:
        rows=list(csv.DictReader(Path(r['saved']).read_text(encoding='utf-8-sig').splitlines()));print('CSV',len(rows),'columns',list(rows[0]),flush=True)
        out={}
        for k in rows[0]:
            if any(t in k.lower() for t in ['tug','age','participant','subject','visit']):out[k]=dict(nonempty=sum(x.get(k,'') not in ['', 'NA','NaN','nan'] for x in rows),unique=len(set(x.get(k,'') for x in rows)))
        save(H/'nmd415-csv-audit.json',dict(rows=len(rows),columns=list(rows[0]),selected_column_counts=out,source_sha256=r['sha256']))
    if r['key']=='synapse_wiki' and 'summary' in r:print('WIKI',str(r['summary'].get('markdown',''))[:7000],flush=True)
    if r['key']=='synapse_headers' and 'summary' in r:print('HEADERS',str(r['summary'])[:4500],flush=True)
