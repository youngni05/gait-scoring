"""Inspect real pose/score linkage and current primary-source access, no fitting."""
from pathlib import Path
import json,csv,hashlib,requests,collections,io
H=Path('D:/Projects/D_P/studies/20261104-camera-tug-data-search')
B=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    out={}
    prior=json.loads((H/'file-audit.json').read_text(encoding='utf8'))['mmu132'];inv={r['name'] for r in prior['files']}
    cp=B/'mmu132-final_gait_feature.csv.download';rp=B/'mmu132-README.md.download'
    for file in [cp,rp]:
        expected=next(r['sha256'] for r in prior['small_files'] if r['file']==str(file))
        assert sha(file)==expected
    rows=list(csv.DictReader(cp.open(encoding='utf-8-sig')));ids=[r['ID'] for r in rows];assert len(ids)==len(set(ids))==125
    joined=[pid for pid in ids if f'MMU-FRiP/Raw/2D/{pid}_2D/{pid}_2D.json' in inv and f'MMU-FRiP/Raw/3D/{pid}_3D/{pid}_3D.json' in inv]
    scores=[float(r['score']) for r in rows];ages=[float(r['age']) for r in rows]
    out['MMU_FRiP']={'pose_people':132,'score_rows':len(rows),'pose_2D_3D_and_score_joined':len(joined),'ages65plus':sum(a>=65 for a in ages),'age_range':[min(ages),max(ages)],'score_range':[min(scores),max(scores)],'score':'JH-FRAT, not TUGseconds orSPPB','raw_RGB':False,'pose_extractors':'AlphaPoseHalpe26;HybrIKSMPL24','capture':'sideview30fps perproviderREADME','label_fields':list(rows[0])[-12:],'score_histogram':dict(collections.Counter(scores)),'table_sha256':sha(cp),'readme_sha256':sha(rp),'no_demographic_or_score_component_inputs_used':True,'status':'125 file-indexlinks verified; no newmodel evaluation'}
    urls={'PDFE_figshare':'https://api.figshare.com/v2/articles/14984667','KINECAL_register':'https://physionet.org/files/kinecal/1.0.3/register.csv','SCAI_access':'https://physionet.org/content/scai-sci-gait-dataset/'}
    for key,url in urls.items():
      try:
        with requests.get(url,stream=True,timeout=35) as r:
            data=b''
            for block in r.iter_content(65536):
                data+=block
                if len(data)>3*1024**2:raise ValueError('metadata cap reached')
            p=B/(key+'.metadata');p.write_bytes(data)
            rec={'url':url,'status':r.status_code,'bytes':len(data),'sha256':sha(p),'path':str(p)}
            if key=='PDFE_figshare' and r.status_code==200:
                m=json.loads(data);rec.update(title=m['title'],license=m.get('license'),files=[{k:x.get(k) for k in ['id','name','size','download_url','computed_md5']} for x in m.get('files',[])],description=m.get('description'))
                for f in rec['files']:
                    if f['size']<2*1024**2 and f['name'].lower().endswith(('.txt','.csv','.xlsx')):
                        q=requests.get(f['download_url'],timeout=35);q.raise_for_status();assert len(q.content)==f['size'];assert hashlib.md5(q.content).hexdigest()==f['computed_md5'];pp=B/('PDFE-'+Path(f['name']).name);pp.write_bytes(q.content);f['local']=str(pp);f['sha256']=sha(pp)
            elif key=='KINECAL_register' and r.status_code==200:
                rr=list(csv.DictReader(io.StringIO(data.decode('utf8-sig'))));rec.update(rows=len(rr),columns=list(rr[0]),first_rows=rr[:2])
            out[key]=rec
      except Exception as e:out[key]={'url':url,'error':repr(e)}
      print('AUDITED',key,flush=True)
    save(H/'visual-function-followup.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
