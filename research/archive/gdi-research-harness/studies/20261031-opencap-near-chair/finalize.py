from study import *
import zlib,platform
def main():
    protocol=json.loads((HERE/'protocol.json').read_text());assert sha(HERE/'study.py')==protocol['code_sha256']
    assert json.loads((HERE/'input-verification.json').read_text())['passed'];v=json.loads((HERE/'results-verification.json').read_text());assert v['passed'] and v['final_models_refit']==200 and v['inner_fits_refit']==2400
    for n,h in protocol['source_hashes'].items():assert sha(HERE.parent/n)==h
    for n,h in protocol['cohorts']['cohort176']['hashes'].items():assert sha(HERE/'cohort176'/n)==h
    entries={r['name']:r for r in json.loads((HERE/'acquisition-plan.json').read_text())['entries']}
    receipts=json.loads((HERE/'acquisition.json').read_text());assert len(receipts)==12
    for r in receipts:
        p=HERE/'raw'/r['name'];b=p.read_bytes();e=entries[r['name']];assert len(b)==e['size'] and zlib.crc32(b)==e['crc'] and sha(p)==r['sha256']
    save('visual-review.json',{'reviewed':True,'file':'quality-review.png','notes':'Four cases inspected; basic acceptance only0.5s in266 and406; marker consistency is not independent ground truth.'})
    import sklearn
    save('environment.json',{'python':sys.version,'numpy':np.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'execution':'local CPU, existing dependencies'})
    artifacts={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'};save('artifact-manifest.json',artifacts)
    for n,h in artifacts.items():assert sha(HERE/n)==h
    print(json.dumps({'passed':True,'files':len(artifacts),'people':176,'raw_marker_files':12,'final_models':200,'inner_fits':2400}),flush=True)
if __name__=='__main__':main()
