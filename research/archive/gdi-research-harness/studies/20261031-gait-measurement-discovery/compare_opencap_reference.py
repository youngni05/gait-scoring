"""Matched speed-summary reference prevents attributing richer speed inputs to joints."""
from common import *
from analyze_comparison import read_predictions,paired_ci


def main():
    result = {}
    hashes = {}
    for name in ['opencap_tug','opencap_patterns_tug']:
        p = read_predictions(name)
        p['error'] = abs(p.prediction-p.target)
        baseline = p[p.variant=='RF|family:speed'].groupby('sample').error.mean().sort_index().to_numpy()
        d,meta = load(name)
        assert len(baseline)==len(d['y'])
        variants = ['selected_family','RF|family:'+'+'.join(meta['groups'])]
        variants += ['RF|family:speed+'+f for f in meta['groups'] if f!='speed']
        contrasts = {}
        for variant in variants:
            g = p[p.variant==variant]
            if len(g):
                errors = g.groupby('sample').error.mean().sort_index().to_numpy()
                contrast = paired_ci(errors-baseline)
                assert abs(contrast['difference']-(errors.mean()-baseline.mean()))<1e-12
                contrasts[variant] = {'MAE':float(errors.mean()),'minus_RF_speed':contrast}
        result[name] = {'RF_speed_MAE':float(baseline.mean()),'comparisons':contrasts}
        hashes.update({str(path):sha(path) for path in (OUT/name).glob('r*f*.predictions.csv.gz')})
    result['provenance'] = {'source_sha256':hashes,'code_sha256':sha(HERE/'compare_opencap_reference.py'),
        'scope':'Post-fitting exploratory contrasts of existing fixed candidates. Same model and same held-out people. Paired intervals conditional on fits; no new learning.'}
    save(HERE/'opencap-speed-reference-comparisons.json',result)
    print('MATCHED_OPENCAP_SPEED_REFERENCE_COMPARISONS_RECALCULATED',flush=True)


if __name__ == '__main__':
    main()
