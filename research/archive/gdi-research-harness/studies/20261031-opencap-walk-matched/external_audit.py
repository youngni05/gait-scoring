from common import *
import re,zipfile,io

def main():
    cells=json.loads((HERE/'external/gait2care-cells.json').read_text(encoding='utf8'))
    visits=[];summ={}
    for sheet in ['Week0','Week8']:
        rows=[]
        for row in cells[sheet][4:]:
            a={re.sub(r'\d','',k):v for k,v in row.items()}
            if not str(a.get('A','')).startswith('GAIT2CARE-'):continue
            record={'id':a['A'],'visit':sheet}
            for col,name in [('T','sts'),('W','tug'),('V','sppb'),('N','clinical_speed'),('Y' if sheet=='Week0' else 'AM','imu_minutes'),('AI' if sheet=='Week0' else 'AW','imu_speed')]:
                try:record[name]=float(a.get(col,''))
                except ValueError:record[name]=np.nan
            rows.append(record)
        df=pd.DataFrame(rows);assert not df.id.duplicated().any();visits.extend(rows)
        valid=(df.imu_speed>0)&(df.imu_minutes>0)
        summ[sheet]={'people':len(df),'imu_summary':int(valid.sum()),'sts_positive':int((df.sts>0).sum()),'tug_positive':int((df.tug>0).sum()),
                     'imu_with_sts':int((valid&(df.sts>0)).sum()),'imu_with_tug':int((valid&(df.tug>0)).sum()),
                     'imu_with_sts20plus':int((valid&(df.sts>=20)).sum()),'imu_with_tug20plus':int((valid&(df.tug>=20)).sum()),
                     'sts_range':[float(df.sts.min()),float(df.sts.max())]}
    visits=pd.DataFrame(visits);visits.to_csv(HERE/'external/gait2care-availability.csv',index=False)
    mask=(visits.imu_speed>0)&(visits.imu_minutes>0)
    summ['unique_people_with_imu_summary']=int(visits.loc[mask,'id'].nunique())
    summ['unique_people_with_imu_and_sts']=int(visits.loc[mask&(visits.sts>0),'id'].nunique())
    summ['unique_people_with_imu_and_tug']=int(visits.loc[mask&(visits.tug>0),'id'].nunique())
    summ['limits']='Workbook acquired and cached source values inspected. Raw IMU archive not acquired. Long-walk aggregates cannot be called a same6s experiment. T4 says Time(seconds)5 squats beneath Chair Stand Test; standardization needs source protocol check.'
    z=zipfile.ZipFile(HERE/'external/plos-s002.zip');tables={};plos={}
    for name in z.namelist():
        if name.endswith('.csv'):
            df=pd.read_csv(io.BytesIO(z.read(name)));assert not df.StudyID.duplicated().any()
            tables[Path(name).stem]=df
            plos[Path(name).stem]={'people':len(df),'columns':df.columns.tolist()}
    def small(name,col):
        df=tables[name].iloc[:,:2].copy();df.columns=['id',col];return df
    joined=small('GaitSpeed_perSub','walk3m_seconds').merge(small('5ChairStand_perSub','sts'),on='id',validate='one_to_one').merge(small('TUG_perSub','tug'),on='id',how='left',validate='one_to_one')
    joined.to_csv(HERE/'external/plos-linked-scores.csv',index=False)
    plos['linked']={'walk_and_sts':len(joined),'walk_sts_tug':int(joined.tug.notna().sum()),'sts20plus':int((joined.sts>=20).sum()),'sts_range':[float(joined.sts.min()),float(joined.sts.max())],
                    'union_people_any_file':len(set().union(*[set(df.StudyID) for df in tables.values()])),'raw_rgb_or_joint_files':0,'limit':'3m clinician stopwatch gait and scores only; no6s motion pattern experiment possible with this release.'}
    save('external-audit.json',{'GAIT2CARE':summ,'PLOS228':plos})
    print(json.dumps({'GAIT2CARE':summ,'PLOS228':plos['linked']},indent=2))
if __name__=='__main__':main()
