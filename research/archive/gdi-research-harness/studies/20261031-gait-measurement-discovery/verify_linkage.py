"""Cross-check all weekly inputs against original provider tables and Excel aggregation."""
from common import *
import openpyxl
import re


def main():
    src = STUDIES/'20261031-mobilised-uncertainty-review/extracted'
    files = sorted((src/'Aggregated').glob('*weekly_agg_filtered.csv'))
    weeks = pd.concat([pd.read_csv(p) for p in files],ignore_index=True).rename(columns={'visit_number':'visit.number'})
    keys = ['participantid','visit.number']
    assert not weeks.duplicated(keys).any()
    checks = {}
    for name in ['mobilise_sppb','mobilise_tug','mobilise_nonwalk']:
        g = pd.read_csv(OUT/f'{name}_linked.csv')
        d,meta = load(name)
        names = meta['features']
        joined = g.merge(weeks[keys+names],on=keys,suffixes=('_input','_original'),validate='one_to_one')
        assert len(joined)==len(g)==len(d['y'])
        for column in names:
            np.testing.assert_allclose(joined[column+'_input'],joined[column+'_original'],atol=0,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(g.participantid.astype(str),d['ids'])
        np.testing.assert_array_equal(g.target,d['y'])
        checks[name] = {'people':len(g),'weekly_fields_exactly_matched':len(names)}
    d,meta = load('older44_sppb')
    path = ROOT/'outputs/gait-public-access-20261015/OlderAdults44/05_gait_parameters.xlsx'
    book = openpyxl.load_workbook(path,read_only=True,data_only=True)
    tuples = list(book.active.values)
    book.close()
    by_person,walks,current = {},set(),None
    for row in tuples[1:]:
        if row[0] is not None:
            current = int(re.search(r'\d+',row[0]).group())
        assert (current,int(row[1])) not in walks
        walks.add((current,int(row[1])))
        v = []
        for j in range(2,28,2):
            v.append((float(row[j])+float(row[j+1]))/2)
        for j in range(2,28,2):
            v.append(abs(float(row[j])-float(row[j+1])))
        by_person.setdefault(str(current),[]).append(v)
    scalar_rebuilt = np.array([[sum(trial[j] for trial in by_person[pid])/5 for j in range(26)] for pid in d['ids']])
    # Vector and scalar floating reductions differ by up to two ULPs at cadence ~100.
    # Keep the original comparison tolerance; use the same vector reduction after independent source mapping.
    rebuilt = np.stack([np.mean(by_person[pid],axis=0) for pid in d['ids']])
    np.testing.assert_allclose(rebuilt,d['x'],atol=1e-14,rtol=0)
    assert len(walks)==220 and all(len(trials)==5 for trials in by_person.values())
    checks['older44_sppb'] = {'people':44,'trials':220,'bilateral_and_asymmetry_features_independently_rebuilt':26,
        'vector_reduction_max_difference':float(abs(rebuilt-d['x']).max()),'scalar_reduction_float_rounding_max_difference':float(abs(scalar_rebuilt-d['x']).max()),'comparison_tolerance_unchanged':1e-14}
    sources = {str(p):sha(p) for p in files+[path]}
    save(HERE/'source-linkage-verification.json',{'passed':True,'checks':checks,'source_sha256':sources,'scope':'Original weekly inputs and supplied Vicon parameters; not independent adjudication of clinical test truth or marker accuracy.'})
    print('ORIGINAL_WEEKLY15_INPUTS_AND_EXCEL26_MEASUREMENTS_VERIFIED',json.dumps(checks),flush=True)


if __name__ == '__main__':
    main()
