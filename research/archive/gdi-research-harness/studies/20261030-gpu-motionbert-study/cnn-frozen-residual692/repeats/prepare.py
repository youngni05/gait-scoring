"""Copy the immutable experiment into two isolated seed-only variants."""
from pathlib import Path
import argparse,hashlib,json,shutil


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--source',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args()
    source=args.source.resolve();out=args.output.resolve()
    manifest=json.loads((source/'manifest.json').read_text())
    assert sha(source/'manifest.json')=='bf76bc5bc73fc9226d907d90026d9f8c5ad0944baf86e21d79448022357475f2'
    for name,digest in manifest.items():assert sha(source/name)==digest,name
    config=json.loads((source/'protocol.json').read_text())
    assert config['seed']==20261028
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in (20261029,20261030):
        dest=out/('seed%d'%seed)
        new=dict(config,seed=seed)
        assert {k:v for k,v in new.items() if k!='seed'}=={k:v for k,v in config.items() if k!='seed'}
        if dest.exists():
            assert json.loads((dest/'protocol.json').read_text())==new
        else:
            dest.mkdir()
            for name in manifest:
                target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
                if name=='protocol.json':save(target,new)
                else:shutil.copyfile(source/name,target)
            changed=dict(manifest,**{'protocol.json':sha(dest/'protocol.json')})
            save(dest/'manifest.json',changed)
        current=json.loads((dest/'manifest.json').read_text())
        assert set(current)==set(manifest)
        for name,digest in current.items():
            assert sha(dest/name)==digest,name
            if name!='protocol.json':assert digest==manifest[name],name
        results.append(dict(seed=seed,directory=str(dest),manifest_sha256=sha(dest/'manifest.json'),only_protocol_seed_changed=True))
    save(out/'preparation.json',dict(source_manifest_sha256=sha(source/'manifest.json'),variants=results))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
