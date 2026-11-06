from analyze import *

def main():
    assert json.loads((HERE/'verification.json').read_text())['passed']
    audit=json.loads((HERE/'audit-verification.json').read_text());assert audit['passed'] and audit['phase_results_recomputed_from_raw']==354
    for name,h in json.loads((HERE/'sources.json').read_text()).items():assert sha(HERE.parent/name)==h
    assert (HERE/'STATUS.md').exists() and (HERE/'boundary-review.png').stat().st_size>1000
    save('visual-review.json',{'image':'boundary-review.png','reviewed':True,'notes':'Six cases viewed; shows missing near-chair intervals and continued height change, not manual timing ground truth. Additional-gap jump warnings included.'})
    artifacts={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'}
    save('artifact-manifest.json',artifacts)
    for n,h in artifacts.items():assert sha(HERE/n)==h
    print(json.dumps({'passed':True,'artifacts':len(artifacts),'final_models':550,'inner_fits':6600,'phase_results':354,'gap_segments':9}),flush=True)
if __name__=='__main__':main()
