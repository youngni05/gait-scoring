"""Download only CP and ITW sagittal gait videos from the public LIVE archive; validate each ZIP CRC."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import struct
import sys
import zlib

ROOT = Path('D:/Projects/D_P')
SOURCE = ROOT / 'gdi-research-harness/studies/20261031-cnn-high-gdi/research'
sys.path.insert(0, str(SOURCE))
from inspect_public_video import fetch  # noqa: E402

HERE = Path(__file__).resolve().parent
BIG = Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
VIDEOS = BIG / 'videos'


def download(row):
    name = row['name']
    group = name.split('/')[1]
    dest = VIDEOS / group / Path(name).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        payload = dest.read_bytes()
        assert len(payload) == row['size'] and zlib.crc32(payload) == row['crc']
    else:
        start = row['header_offset']
        blob, cr = fetch(str(start) + '-' + str(start + row['compressed_size'] + 4095))
        assert int(cr.split()[1].split('-')[0]) == start
        header = struct.unpack('<4s5H3L2H', blob[:30])
        assert header[0] == b'PK\x03\x04'
        n, e = header[-2:]
        assert blob[30:30 + n].decode('utf-8') == name
        raw = blob[30 + n + e:30 + n + e + row['compressed_size']]
        assert row['compression'] == 8
        payload = zlib.decompress(raw, -15)
        assert len(payload) == row['size'] and zlib.crc32(payload) == row['crc']
        dest.write_bytes(payload)
    return dict(file=str(dest.relative_to(VIDEOS)).replace('\\', '/'), group=group, person=name.split('/')[2],
                size=len(payload), sha256=hashlib.sha256(payload).hexdigest(), zip_crc_verified=True)


def main():
    inventory = json.loads((SOURCE / 'video_inventory.json').read_text())
    rows = [r for r in inventory['files']
            if r['name'].startswith(('VIDEO/CP/', 'VIDEO/ITW/')) and r['name'].upper().endswith('-SAGITTAL.MOV') and '-GAIT-' in r['name'].upper()]
    print('expected', len(rows), round(sum(r['size'] for r in rows) / 1e9, 2), 'GB', flush=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download, r): r for r in rows}
        for f in as_completed(futures):
            out.append(f.result())
            if len(out) % 20 == 0:
                print('done', len(out), flush=True)
    assert len(out) == len(rows)
    out.sort(key=lambda r: r['file'])
    (HERE / 'download.json').write_text(json.dumps(dict(source=inventory['url'], videos=out), indent=1), encoding='utf-8')
    print('complete', len(out), flush=True)


if __name__ == '__main__':
    main()
