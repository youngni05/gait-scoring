"""Confirm public archive size, person/visit joins, and four small source files."""
from pathlib import Path
import csv,json,re,math,collections,hashlib,requests,io,zipfile,datetime
H=Path('D:/Projects/D_P/studies/20261104-camera-tug-data-search');B=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def finite(value):
    try:return math.isfinite(float(value))
    except (ValueError,TypeError):return False
def main():
    inventory=read(B/'nmd415-zip-inventory.json');labels=list(csv.DictReader((B/'new_opencap_csv.txt').read_text(encoding='utf-8-sig').splitlines()));table={(r['subid'],str(int(float(r['visit'])))):r for r in labels};assert len(table)==len(labels)
    by={}
    for r in inventory:
        m=re.match(r'Neuromuscular_OpenCap_Dataset/(sub-\d+)/visit-([^/]+)/ses-(\d+)/(Markers|Kinematics)/.+_task-(.+)\.(trc|mot)$',r['name'])
        if m:
            pid,visit,ses,kind,task,extension=m.groups();by.setdefault((pid,visit,ses),{})[(task,extension)]=r
    complete=[];tug=[]
    for (pid,visit,ses),items in by.items():
        row=table.get((pid,visit))
        if not row or not finite(row['tft_tug']) or float(row['tft_tug'])<=0:continue
        common=dict(person=pid,visit=visit,session=ses,age=float(row['age']) if finite(row['age']) else None,diagnosis=row['diag'],tug=float(row['tft_tug']),split=row['split'])
        if ('10mwt','trc') in items and ('10mwt','mot') in items:complete.append({**common,'markers':items['10mwt','trc']['name'],'kinematics':items['10mwt','mot']['name']})
        for task in ['tug_cone','tug_line']:
            if (task,'trc') in items and (task,'mot') in items:tug.append({**common,'task':task,'markers':items[task,'trc']['name'],'kinematics':items[task,'mot']['name']})
    def summary(rows):
        unique={}
        for r in sorted(rows,key=lambda r:(r['person'],int(r['visit']),int(r['session']))):unique.setdefault(r['person'],r)
        values=list(unique.values());ages=[x for x in values if x['age'] is not None and x['age']>=65]
        return dict(sessions=len(rows),visits=len(set((x['person'],x['visit']) for x in rows)),people=len(values),people65plus=len(ages),first_eligible_per_person_diagnoses=dict(collections.Counter(x['diagnosis'] for x in values)),older_first_eligible_diagnoses=dict(collections.Counter(x['diagnosis'] for x in ages)),first_eligible_TUG_bands=dict(under12=sum(x['tug']<12 for x in values),from12to20=sum(12<=x['tug']<20 for x in values),atleast20=sum(x['tug']>=20 for x in values)))
    size=dict(archive_bytes=6584902803,archive_GB=6.584902803,archive_GiB=6584902803/1024**3,uncompressed_bytes=sum(r['size'] for r in inventory),uncompressed_GB=sum(r['size'] for r in inventory)/1e9,files=sum(r['size']>0 for r in inventory),members_including_folders=len(inventory))
    out=dict(status='file_inventory_and_label_join_passed',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source='https://zenodo.org/records/22309771',source_inventory_sha256=sha((B/'nmd415-zip-inventory.json').read_bytes()),labels_sha256=sha((B/'new_opencap_csv.txt').read_bytes()),size=size,ten_meter_walk_and_TUG_label=summary(complete),TUG_motion_and_TUG_label=summary(tug),join_definition='Exactperson+visitmatches;requirebothmarker.trcandkinematics.motofsame10mwt/session. Countsseparatelyreportparticipants,visits,sessions. Age/bandsusefirsteligiblevisit/sessionperperson,notanymostfavorablevisit. TUGcone/line tasks maybothmaptosingleclinicianTUGcolumn;exactpertrialtiminglinknotestablished.',video_files=0,limits='OpenCap-derived3Destimates,notsensor-measured3DandnotrawRGB. Inventoryjoinprovesfilenamesandlabelavailability,notsignalquality,phaseboundaries,exactTUGtriallabelorphoneaccuracy. No fullarchive download or new training.',available_exact_cohort=complete)
    (H/'opencap-size-and-join.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    # Pilot: read one matched walk's TRC/MOT, one TUG MOT, and session metadata.
    sample=complete[0];items=by[sample['person'],sample['visit'],sample['session']];names=[sample['markers'],sample['kinematics']];tn=next((r['kinematics'] for r in tug if all(r[k]==sample[k] for k in ['person','visit','session'])),None)
    if tn:names.append(tn)
    meta=next((r['name'] for r in inventory if f"/{sample['person']}/visit-{sample['visit']}/ses-{sample['session']}/" in r['name'] and r['name'].endswith('.yaml')),None)
    if meta:names.append(meta)
    SIZE=size['archive_bytes'];URL='https://zenodo.org/records/22309771/files/Neuromuscular_OpenCap_Dataset.zip?download=1'
    class Remote(io.RawIOBase):
        def __init__(self):self.pos=0;self.received=0
        def tell(self):return self.pos
        def seek(self,n,w=0):self.pos=n if w==0 else self.pos+n if w==1 else SIZE+n;return self.pos
        def read(self,n=-1):
            if n<0:n=SIZE-self.pos
            n=min(n,SIZE-self.pos)
            if n<=0:return b''
            assert n<=8*1024**2 and self.received+n<20*1024**2
            with requests.get(URL,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}','Accept-Encoding':'identity'},stream=True,timeout=40) as response:
                assert response.status_code==206,'Rangeignored,fullarchivenotdownloaded';body=response.raw.read(n+1);assert len(body)==n;self.pos+=n;self.received+=n;return body
    remote=Remote();z=zipfile.ZipFile(remote);dest=B/'opencap-file-pilot';dest.mkdir(exist_ok=True);records=[]
    for name in names:
        body=z.read(name);path=dest/Path(name).name;path.write_bytes(body);lines=body.decode('utf8').splitlines();r=dict(member=name,file=str(path),bytes=len(body),sha256=sha(body),zip_CRC_checked=True)
        if name.endswith('.trc'):r['header']=lines[:5]
        if name.endswith('.mot'):r['header']=lines[:9]
        if name.endswith('.yaml'):r['text']=body.decode('utf8')
        records.append(r)
    pilot=dict(status='four_small_files_downloaded_and_CRC_verified',range_bytes_received=remote.received,uncompressed_pilot_bytes=sum(r['bytes'] for r in records),records=records,whole_archive_downloaded=False,clinical_model_fitted=False);(H/'opencap-file-pilot.json').write_text(json.dumps(pilot,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k not in ['available_exact_cohort']},indent=2));print('PILOT',len(records),'rangebytes',remote.received)
if __name__=='__main__':main()
