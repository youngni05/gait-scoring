"""Acquire only public files into this study; validate archived source hashes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / 'raw'
RAW.mkdir(exist_ok=True)

def fetch(url, path, expected=None):
    if not path.exists():
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/octet-stream,*/*', 'Referer': 'https://data.mendeley.com/datasets/xgw6bg3g8h/1'})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        path.write_bytes(data)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected:
        assert digest == expected, path
    return {'file': path.name, 'bytes': len(data), 'sha256': digest, 'url': url}

if sys.argv[1] == 'rgb':
    receipts = []
    for name in ['README.md', 'FullReadMe.txt', 'exampleSPPB.m', 'walkingSet.mat', 'balanceSet.mat', 'ststandSet.mat']:
        receipts.append(fetch('https://raw.githubusercontent.com/ispstiima/SPPBdataset/main/' + name, RAW / name))
        print(name, receipts[-1]['bytes'], flush=True)
    (HERE / 'rgb-source-receipt.json').write_text(json.dumps(receipts, indent=2))
elif sys.argv[1] == 'markers':
    metadata = json.loads((ROOT / 'outputs/gait-public-access-20261015/older44-metadata.json').read_text(encoding='utf-8-sig'))
    selected = [f for f in metadata['files'] if f['filename'].endswith('.csv')]
    def one(f):
        c = f['content_details']
        record = fetch(c['download_url'], RAW / f['filename'], c['sha256_hash'])
        print(record['file'], record['bytes'], flush=True)
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(one, selected))
    (HERE / 'marker-source-receipt.json').write_text(json.dumps(receipts, indent=2))
