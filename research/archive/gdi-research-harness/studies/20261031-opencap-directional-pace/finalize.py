from study import *

def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert protocol['code_sha256']==sha(HERE/'study.py')
    assert json.loads((HERE/'input-verification.json').read_text())['passed']
    verified=json.loads((HERE/'results-verification.json').read_text());assert verified['passed'] and verified['models']==450 and verified['inner_fits']==5400
    cases=pd.read_csv(HERE/'case-results.csv');diag=pd.read_csv(HERE/'diagnostic-subsets.csv')
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort
        for n,h in info['hashes'].items():assert sha(out/n)==h
        p=pd.read_csv(out/'predictions.csv');persons=pd.read_csv(out/'person-results.csv')
        for r in cases[cases.cohort==cohort].itertuples():
            g=p[(p.view==r.view)&(p.id==r.id)];e=g.prediction-g.target
            actual=[g.prediction.mean(),g.prediction.std(ddof=0),g.prediction.min(),g.prediction.max(),abs(e).mean(),np.maximum(-e,0).mean()]
            assert np.allclose(actual,[r.prediction,r.sd,r.minimum,r.maximum,r.MAE,r.underprediction],atol=1e-10,rtol=0)
        for r in diag[diag.cohort==cohort].itertuples():
            g=persons[persons.view==r.view]
            if r.subset=='long_without397':g=g[(g.target>=20)&(g.id!='sub-397')]
            elif r.subset=='common176':g=g[g.id!='sub-283']
            else:g=g[(g.target>=20)&(g.id!='sub-283')]
            assert len(g)==r.people
            for k in ['MAE','underprediction','overprediction']:assert abs(g[k].mean()-getattr(r,k))<1e-10
    for n,h in protocol['source_hashes'].items():assert sha(HERE.parent/n)==h
    assert (HERE/'STATUS.md').exists() and (HERE/'comparison.png').stat().st_size>1000
    save('completion-verification.json',{'passed':True,'models':450,'inner_fits':5400,'case_stability_recomputed':True,'diagnostic_subsets_recomputed':True,'source_and_input_hashes_unchanged':True})
    artifacts={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'}
    save('artifact-manifest.json',{'files':artifacts})
    for n,h in artifacts.items():assert sha(HERE/n)==h
    print(json.dumps({'passed':True,'models':450,'inner_fits':5400,'artifacts':len(artifacts)}),flush=True)
if __name__=='__main__':main()
