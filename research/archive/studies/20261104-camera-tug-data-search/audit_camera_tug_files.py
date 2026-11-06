from pathlib import Path
import sys,requests,json,csv,zipfile,io,collections,urllib.parse,hashlib,xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf8')
H=Path('D:/Projects/D_P/studies/20261104-camera-tug-data-search');B=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104')
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
def sha(b):return hashlib.sha256(b).hexdigest()
def small(url,key,cap=5*1024**2):
    with requests.get(url,stream=True,timeout=35) as r:
        chunks=[];n=0
        for c in r.iter_content(65536):
            n+=len(c)
            if n>cap:raise ValueError('small-file cap reached; no large archive downloaded')
            chunks.append(c)
        body=b''.join(chunks);p=B/key;p.write_bytes(body);return dict(url=url,status=r.status_code,size=len(body),sha256=sha(body),content_type=r.headers.get('Content-Type'),file=str(p)),body
report={}
for slug,label in [('limzhekhae1008/mmu-frip','mmu132'),('chenhui0331/mmu-parkinsons-disease-tug-test-pose-dataset','mmupd'),('insyirahazman/parkinsons-disease-dataset','pd')]:
    files=[];token=None;seen=set();pages=[]
    for page in range(35):
        url='https://www.kaggle.com/api/v1/datasets/list/'+slug;params=dict(pageSize=1000)
        if token:params['pageToken']=token
        r=requests.get(url,params=params,timeout=35);assert r.status_code==200;rj=r.json();pages.append(dict(url=r.url,status=r.status_code,sha256=sha(r.content)));files.extend(rj.get('datasetFiles',[]));token=rj.get('nextPageToken')
        if not token:break
        assert token not in seen,'Pagination did not advance';seen.add(token)
    unique={x['name']:x for x in files};summary=dict(slug=slug,files=list(unique.values()),pages=pages,complete=not bool(token),extensions=dict(collections.Counter(Path(x).suffix for x in unique)),participant_codes=len(set(m for name in unique for m in __import__('re').findall(r'P\d{3}',name))));report[label]=summary;print(label,summary['complete'],len(unique),summary['extensions'],summary['participant_codes'],flush=True)
    for name in unique:
        if name.endswith('final_gait_feature.csv') or name.endswith('README.md'):
            url='https://www.kaggle.com/api/v1/datasets/download/'+slug+'/'+urllib.parse.quote(name,safe='')
            try:
                meta,body=small(url,label+'-'+Path(name).name+'.download');summary.setdefault('small_files',[]).append(meta)
                if body.startswith(b'PK'):
                    z=zipfile.ZipFile(io.BytesIO(body));summary.setdefault('small_zip_members',[]).extend(z.namelist());body=z.read(z.namelist()[0])
                if name.endswith('.csv') and meta['status']==200:
                    rows=list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))));summary['feature_csv']=dict(rows=len(rows),columns=list(rows[0]),source_download_sha256=meta['sha256']);print('MMUCSV',len(rows),list(rows[0]),flush=True)
                elif name.endswith('README.md') and meta['status']==200:summary['readme']=body.decode('utf8',errors='replace');print(summary['readme'][:12000],flush=True)
            except Exception as e:summary.setdefault('small_errors',[]).append(dict(name=name,error=repr(e)))
save(H/'kaggle-file-inventory.json',report)
for key,ident in [('sail_labels','1Xzrgkezc5NdCrjYNWprW3hpoPZXeXBTo'),('sail_video_link','1oX0PrtOPAGcuX9Hg0ZLWcS1EuKpCOXcE')]:
    try:
        meta,body=small('https://drive.google.com/uc?export=download&id='+ident,key+'.bin');report[key]=meta
        if key=='sail_labels' and body.startswith(b'PK'):
            z=zipfile.ZipFile(io.BytesIO(body));ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};strings=[]
            if 'xl/sharedStrings.xml' in z.namelist():strings=[''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
            sheets={}
            for name in z.namelist():
                if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
                    root=ET.fromstring(z.read(name));rows=[]
                    for row in root.findall('m:sheetData/m:row',ns):
                        vals=[]
                        for c in row.findall('m:c',ns):
                            v=c.find('m:v',ns);text=v.text if v is not None else ''.join(c.itertext());vals.append(strings[int(text)] if c.attrib.get('t')=='s' and text else text)
                        rows.append(vals)
                    sheets[name]=dict(row_count=len(rows),first_rows=rows[:3])
            report[key]['sheets']=sheets
        print(key,report[key],flush=True)
    except Exception as e:report[key]=dict(error=repr(e));print(key,e,flush=True)
# Anonymous project metadata listing, never accept terms/register a user account.
try:
    r=requests.post('https://repo-prod.prod.sagebase.org/repo/v1/entity/children',json=dict(parentId='syn52540892',includeTypes=['folder','file','table','link']),timeout=35);report['synapse_children']=dict(status=r.status_code,data=r.json());print('synapsechildren',str(report['synapse_children'])[:5000],flush=True)
except Exception as e:report['synapse_children']=dict(error=repr(e))
save(H/'file-audit.json',report)
