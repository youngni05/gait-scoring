"""Independent arithmetic/provenance verification, not clinical label validation."""
from common import *
import io,zlib,zipfile

def main():
    d=load();summary=json.loads((HERE/'matching-summary.json').read_text());man=pd.read_csv(PRIOR/'STS_common6/manifest.csv')
    for p,h in summary['sources'].items():assert sha(p)==h
    assert sha(HERE/'protocol.json')==summary['protocol_sha256']
    prior_preparation=json.loads((PRIOR/'preparation.json').read_text())
    assert sha(PRIOR/'STS_common6/splits.json')==prior_preparation['cohorts']['STS_common6']['split_sha256']
    assert len(set(d['ids']))==179 and 'sub-283' not in d['ids'] and d['x'].shape==(179,19,181)
    labels=pd.read_csv(LABELS).set_index(['subid','visit']);assert labels.index.is_unique
    channels=['pelvis_tilt','pelvis_list','hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r','subtalar_angle_r','hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l','subtalar_angle_l','lumbar_extension','lumbar_bending','lumbar_rotation']
    for i,r in enumerate(man.itertuples()):
        assert r.id==d['ids'][i] and r.seconds==6 and r.label==labels.loc[(r.id,r.visit),'tft_5xsts']==d['y'][i]
        raw=SOURCE/'raw'/r.file;assert sha(raw)==r.source_sha256
        b=pd.read_csv(io.StringIO(raw.read_text().split('endheader')[1]),sep=r'\s+')
        t=b.time.to_numpy();start=int(r.start_frame);end=np.searchsorted(t,t[start]+6);b=b.iloc[start:end+1];grid=t[start]+np.arange(181)/30
        angles=np.array([np.interp(grid,b.time,b[c]) for c in channels]);pos=np.array([np.interp(grid,b.time,b[c]) for c in ['pelvis_tx','pelvis_ty','pelvis_tz']])
        v=np.gradient(pos,1/30,axis=1);rebuilt=np.vstack([angles,np.hypot(v[0],v[2]),v[1]]).astype(np.float32)
        np.testing.assert_allclose(rebuilt,d['x'][i],rtol=1e-5,atol=1e-5)
    a=pd.read_csv(HERE/'all-pairs.csv');scale=np.nanquantile(d['pattern'],.75,axis=0)-np.nanquantile(d['pattern'],.25,axis=0);scale[scale<1e-8]=1
    np.testing.assert_allclose(scale,summary['iqr_scale'])
    valid=np.isfinite(d['pattern']).all(1);q=pd.read_csv(QUALITY/'quality-audit.csv').query("dataset=='STS_common6'").set_index('id').loc[d['ids']]
    assert len(a)==179*178//2 and not a.duplicated(['i','j']).any()
    for r in a.itertuples():
        i,j=int(r.i),int(r.j);assert 0<=i<j<179 and (r.id_a,r.id_b)==(d['ids'][i],d['ids'][j])
        assert abs(r.sts_gap-abs(d['y'][i]-d['y'][j]))<1e-10
        if valid[i] and valid[j]:
            dz=(d['pattern'][i]-d['pattern'][j])/scale
            expected=np.sqrt(np.dot(dz[:3],dz[:3])/6+np.dot(dz[3:],dz[3:])/28)
            assert abs(r.feature_distance-expected)<1e-10
        assert r.primary==(bool(valid[i]&valid[j]) and r.speed_difference<=.1 and r.feature_distance<=.5 and bool(q.iloc[i].stable and q.iloc[j].stable))
    selected=pd.read_csv(HERE/'selected-pairs.csv');used=set();expected=[]
    for group,limit in [('primary',3),('flagged',2)]:
        mask=a.primary if group=='primary' else a.matched&~a.both_stable
        n=0
        for r in a[mask&a.discordant].sort_values(['sts_gap','feature_distance'],ascending=[False,True]).itertuples():
            if r.id_a in used or r.id_b in used:continue
            expected.append((r.id_a,r.id_b));used.update([r.id_a,r.id_b]);n+=1
            if n==limit:break
    assert list(zip(selected.id_a,selected.id_b))==expected
    index={r['name']:r for r in json.loads((SOURCE/'archive-index.json').read_text())};receipts=json.loads((HERE/'acquisition.json').read_text())
    assert len(receipts)==20
    for r in receipts:
        p=HERE/'raw'/r['name'];b=p.read_bytes();ref=index[r['name']]
        assert len(b)==r['bytes']==ref['size'] and zlib.crc32(b)==ref['crc'] and sha(p)==r['sha256']
    waves=np.load(HERE/'cycle-waveforms.npz');wm=pd.read_csv(HERE/'pair-waveform-differences.csv');names=['r_hip_flexion','r_hip_adduction','r_hip_rotation','r_knee','r_ankle','l_hip_flexion','l_hip_adduction','l_hip_rotation','l_knee','l_ankle']
    for row in selected.itertuples():
        delta=waves['waves'][row.i]-waves['waves'][row.j]
        for c,name in enumerate(names):
            got=wm[(wm['case']==row.case)&(wm.channel==name)].iloc[0]
            np.testing.assert_allclose(got.shape_only_rmse,np.std(delta[c]),atol=1e-10)
    provenance=pd.read_csv(HERE/'label-provenance-all.csv').fillna('');assert len(provenance)==179 and provenance.five_available.sum()==168
    for r in provenance.itertuples():
        actual=[n for n in index if f'/{r.id}/visit-{r.visit}/' in n and n.endswith('.mot') and 'xsts' in n]
        assert set(r.chair_files.split(';') if r.chair_files else [])==set(actual)
    acquisition=json.loads((HERE/'external-acquisition.json').read_text())
    for name,r in acquisition.items():assert sha(HERE/'external'/name)==r['sha256']
    z=zipfile.ZipFile(HERE/'external/plos-s002.zip');ids=[]
    for name in ['GaitSpeed','5ChairStand','TUG']:
        table=pd.read_csv(io.BytesIO(z.read(f'S1 Dataset/{name}_perSub.csv')));ids.append(set(table.StudyID))
    assert len(ids[0]&ids[1])==211 and len(set.intersection(*ids))==210
    # Count source worksheet cells directly, independently of extracted CSV/JSON.
    import xml.etree.ElementTree as ET
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    z=zipfile.ZipFile(HERE/'external/Gait2Care_DDBB.xlsx');totals=[]
    for file,speedcol in [('xl/worksheets/sheet1.xml','AI'),('xl/worksheets/sheet3.xml','AW')]:
        counts=[0,0,0]
        for row in ET.fromstring(z.read(file)).findall('.//m:sheetData/m:row',ns):
            rn=int(row.attrib['r'])
            if rn<5:continue
            vals={c.attrib['r']:c.find('m:v',ns) for c in row.findall('m:c',ns)}
            def number(col):
                v=vals.get(col+str(rn))
                return float(v.text) if v is not None else float('nan')
            if number(speedcol)>0:
                counts[0]+=1;counts[1]+=number('T')>0;counts[2]+=number('W')>0
        totals.append(counts)
    assert totals==[[93,83,84],[93,86,85]],totals
    save('verification.json',{'passed':True,'scope':'computational and source consistency only; no clinical stopwatch or independent tracking truth',
         'people_and_exact6s_windows_rebuilt':179,'pair_distances_checked':len(a),'selection_reproduced':True,'download_crc_sha_files':20,
         'label_task_index_checked_people':179,'waveform_differences_checked':50,'external_download_hashes':len(acquisition),'plos_person_intersections_verified':True,
         'original_splits_hash_verified':True,'gait2care_raw_worksheet_counts_verified':totals,'new_models_trained':0,
         'artifact_sha256':{str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.csv','.npz'] and '__pycache__' not in str(p) and p.name!='verification.json'}})
    print('VERIFIED 179 windows, 15931 pair distances, 20 downloads, 50 waveform differences; no clinical-truth claim')
if __name__=='__main__':main()
