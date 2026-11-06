"""Inspect tutorial source inputs; discard embedded media and never execute it."""
import hashlib,io,json,urllib.request,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent/'public-source'
tree=json.loads((HERE/'tree.json').read_text(encoding='utf-8'))
records=[]
for name in ['MoCap/tutorial_python.ipynb','MoCap/tutorial_MATLAB.mlx']:
    item=next(x for x in tree['tree'] if x['path']==name)
    url=f"https://raw.githubusercontent.com/saeed1262/MoVi-Toolbox/{tree['sha']}/{name}"
    req=urllib.request.Request(url,headers={'User-Agent':'MoVi-local-source-input-audit'})
    with urllib.request.urlopen(req,timeout=60) as response:data=response.read(item['size']+1)
    assert len(data)==item['size']
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==item['sha']
    if name.endswith('.ipynb'):
        notebook=json.loads(data);parts=[]
        for i,cell in enumerate(notebook['cells']):
            parts.append(f"CELL {i} {cell['cell_type']}\n"+''.join(cell.get('source',[])))
        source='\n\n'.join(parts);target=HERE/'tutorial_python.inputs.txt'
        info={'cells':len(notebook['cells']),'embedded_outputs_stored':False}
    else:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            source=archive.read('matlab/document.xml').decode('utf-8')
        target=HERE/'tutorial_MATLAB.document.xml';info={'embedded_outputs_stored':False}
    target.write_text(source,encoding='utf-8')
    records.append({'url':url,'original_git_blob_sha1':blob,'original_bytes_read_in_memory':len(data),
                    'stored_source_file':target.name,'stored_source_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),**info})
    print('Inspected input source',name,'stored',target.stat().st_size,'bytes',flush=True)
(HERE/'tutorial-inputs-manifest.json').write_text(json.dumps({'repo_commit':tree['sha'],'files':records},indent=2),encoding='utf-8')
