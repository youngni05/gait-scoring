"""Public download diagnostics; never print returned auth/error response bodies."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
OUT=HERE/'followup-access'
OUT.mkdir(exist_ok=True)
meta=json.loads((HERE/'raw/current-file.json').read_text(encoding='utf-8-sig'))
base=meta['content_details']['download_url']
urls=[base,base.replace('file_downloaded','file_viewed'),base+'?download=1&check='+str(int(time.time()))]
old=json.loads((HERE.parents[2]/'outputs/gait-public-access-20261015/older44-metadata.json').read_text(encoding='utf-8-sig'))
csvmeta=next(f for f in old['files'] if f['filename']=='Subject01_01.csv')
urls += [csvmeta['content_details']['download_url'],csvmeta['content_details']['view_url']]

def inspect(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':'application/octet-stream,*/*','Referer':'https://data.mendeley.com/datasets/xgw6bg3g8h/1'})
    try:
        with urllib.request.urlopen(req,timeout=35) as response:
            data=response.read()
            result={'url':url,'status':response.status,'content_type':response.headers.get('Content-Type'),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        if result['sha256'] in [meta['content_details']['sha256_hash'],csvmeta['content_details']['sha256_hash']]:
            target=meta['filename'] if result['sha256']==meta['content_details']['sha256_hash'] else csvmeta['filename']
            (OUT/target).write_bytes(data)
            result['verified_file']=target
        return result
    except urllib.error.HTTPError as error:
        return {'url':url,'status':error.code,'error':'HTTP error; body deliberately not logged'}
    except Exception as error:
        return {'url':url,'error':type(error).__name__}

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(inspect,urls))
receipt={'checked_utc':datetime.now(timezone.utc).isoformat(),'results':results}
(OUT/'access-recheck.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
