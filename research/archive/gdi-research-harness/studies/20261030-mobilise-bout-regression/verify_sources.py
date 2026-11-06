"""Independent stdlib parsing checks sampled bouts against all original CSVs."""
from pathlib import Path
import csv,io,json,zipfile,math,hashlib
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parents[2]/'outputs/sppb-tug-candidates-20261030'
features=['averagecadence','averagestrideduration','averagestridespeed','averagestridelength']
expected={}
with (HERE/'selected-bouts.csv').open(newline='',encoding='utf8') as f:
    for r in csv.DictReader(f):
        key=(r['participantid'],r['visit_number'],r['wb_id'])
        values={n:float(r[n]) for n in features+['duration','wbday','starttimestamputc','turn_number_so']}
        if key in expected:assert values==expected[key]
        expected[key]=values
found=set();scanned=0
with zipfile.ZipFile(SOURCE/'mobilise-bouts.zip') as outer:
    for name in outer.namelist():
        if not name.endswith('.zip'):continue
        with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
            with io.TextIOWrapper(inner.open(inner.namelist()[0]),encoding='utf8',newline='') as f:
                reader=csv.reader(f);header=next(reader);ix={n:i for i,n in enumerate(header)}
                for r in reader:
                    if not r:continue
                    scanned+=1
                    key=(r[ix['participantid']],r[ix['visit_number']],r[ix['wb_id']])
                    if key not in expected:continue
                    assert key not in found
                    assert r[ix['datatype']]=='wb'
                    assert all(r[ix[n+'_flag']].lower()=='true' for n in features+['duration'])
                    for n,value in expected[key].items():
                        assert math.isclose(float(r[ix[n]]),value,rel_tol=1e-12,abs_tol=1e-12),(key,n)
                    found.add(key)
        print('Source verified',name,'matched',len(found),flush=True)
assert found==set(expected)
assert scanned==16055401
result={'passed':True,'independent_parser':'stdlib csv.reader','all_raw_rows_scanned':scanned,'unique_selected_bouts_verified':len(found),'all_selected_values_and_quality_flags_match_original':True,'zip_crc_checked_by_reading_all_csvs':True}
(HERE/'source-verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2),flush=True)
