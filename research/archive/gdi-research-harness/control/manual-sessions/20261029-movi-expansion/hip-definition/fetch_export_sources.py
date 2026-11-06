"""Read-only public-source snapshot; no downloaded code is executed."""
import hashlib,json,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'public-source';OUT.mkdir(exist_ok=True)
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'MoVi-local-research-audit','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()
repo='https://api.github.com/repos/saeed1262/MoVi-Toolbox'
tree=json.loads(get(repo+'/git/trees/master?recursive=1'))
(OUT/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
records=[]
for item in tree['tree']:
    p=item['path']
    if item['type']!='blob' or not p.endswith(('.m','.py','.md','.txt')):continue
    assert item['size']<100_000
    url=f"https://raw.githubusercontent.com/saeed1262/MoVi-Toolbox/{tree['sha']}/{p}"
    data=get(url)
    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert actual==item['sha']
    target=OUT/p;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(data)
    records.append({'path':p,'git_blob_sha1':actual,'sha256':hashlib.sha256(data).hexdigest(),'url':url,'bytes':len(data)})
for name,query in [('issues','/issues?state=all&per_page=100'),('commits','/commits?per_page=100'),('branches','/branches?per_page=100')]:
    data=get(repo+query);(OUT/(name+'.json')).write_bytes(data)
    if name=='issues':
        issues=json.loads(data)
        for issue in issues:
            if issue.get('comments',0):
                comments=get(issue['comments_url']);(OUT/f"issue-{issue['number']}-comments.json").write_bytes(comments)
(OUT/'manifest.json').write_text(json.dumps({'repo_commit':tree['sha'],'files':records},indent=2),encoding='utf-8')
print('snapshotted',len(records),'text files',sum(r['bytes'] for r in records),'bytes; commit',tree['sha'])
