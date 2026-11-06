"""Small official-source metadata audit; no bulk datasets or agreements submitted."""
from pathlib import Path
import json,hashlib,urllib.request,concurrent.futures,sys
H=Path(__file__).resolve().parent;B=Path('D:/gait-research-data/D_P/planar-measurement-validation-20261105')/'source-metadata'
URLS={
 'stenum-repository':'https://api.github.com/repos/janstenum/GaitAnalysis-PoseEstimation',
 'stenum-branches':'https://api.github.com/repos/janstenum/GaitAnalysis-PoseEstimation/branches',
 'gpjatk':'https://zenodo.org/api/records/15130800',
 'dual21':'https://zenodo.org/api/records/19720803',
 'movi':'https://zenodo.org/api/records/4022252',
 'older-exercise':'https://rdr.kuleuven.be/api/datasets/:persistentId/?persistentId=doi:10.48804/VNM5WN',
 'gpjatk-agreement':'https://zenodo.org/records/15130800/files/GPJATK_Release_Agreement.pdf?download=1',
 'pd-change':'https://zenodo.org/api/records/22097474',
 'stenum-tree':'https://api.github.com/repos/janstenum/GaitAnalysis-PoseEstimation/git/trees/316916d8a7f22892714c31d044d7e127f0d9937f?recursive=1',
 'stenum-multiple-tree':'https://api.github.com/repos/janstenum/GaitAnalysis-PoseEstimation/git/trees/d3a18a3cf08901adb3cff546972e05251c92d9ac?recursive=1'}

def fetch(item):
    name,url=item
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'D_P research-source-audit/1.0'})
        with urllib.request.urlopen(req,timeout=45) as response:
            data=response.read(5_000_001);assert len(data)<=5_000_000;status=response.status;ctype=response.headers.get('Content-Type')
        path=B/(name+('.pdf' if name.endswith('agreement') else '.json'));path.write_bytes(data)
        return {'name':name,'url':url,'status':status,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'path':str(path),'content_type':ctype}
    except Exception as exc:return {'name':name,'url':url,'status':'failed','reason':str(exc)}

def main():
    B.mkdir(parents=True,exist_ok=True)
    selected={key:value for key,value in URLS.items() if not sys.argv[1:] or key in sys.argv[1:]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,selected.items()))
    out={'metadata_only':True,'no_contracts_or_messages_submitted':True,'records':records}
    destination=H/('source-acquisition-extra.json' if sys.argv[1:] else 'source-acquisition.json');destination.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    for record in records:
        print(record['name'],record['status'],record.get('bytes',record.get('reason')),flush=True)
        if record['status']==200 and record['name'] in ['gpjatk','dual21','movi','pd-change']:
            data=json.loads(Path(record['path']).read_text());print(json.dumps({'title':data.get('metadata',{}).get('title'),'license':data.get('metadata',{}).get('license'),'files':[(r['key'],r['size']) for r in data.get('files',[])]},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
