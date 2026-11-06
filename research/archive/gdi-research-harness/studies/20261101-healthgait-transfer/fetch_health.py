"""Select only public pose/CSV members from the official split ZIP; Python 3.8+."""
import argparse
import concurrent.futures
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import struct
import time
import urllib.request
import zlib
import zipfile

BASE = 'https://zenodo.org/records/14039922/files/'
PART = 1048576000
LAST = 496447815
RANGE_LIMIT = int(os.environ.get('HEALTHGAIT_RANGE_BYTES', str(512*1024)))
ARCHIVE_MEMBERS = os.environ.get('HEALTHGAIT_ARCHIVE_MEMBERS') == '1'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf8')


def fetch(disk, start, length):
    if length > RANGE_LIMIT:
        return b''.join(fetch(disk,start+a,min(RANGE_LIMIT,length-a)) for a in range(0,length,RANGE_LIMIT))
    name = 'Health_Gait.zip' if disk == 25 else 'Health_Gait.z%02d' % (disk + 1)
    size = LAST if disk == 25 else PART
    if start + length > size:
        n = size - start
        return fetch(disk, start, n) + fetch(disk + 1, 0, length - n)
    for attempt in range(5):
        try:
            req = urllib.request.Request(BASE + name + '?download=1&range=%d-%d' % (start, length), headers={'Range': 'bytes=%d-%d' % (start, start + length - 1)})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206 or r.headers.get('Content-Range') != 'bytes %d-%d/%d' % (start, start+length-1, size):
                    raise ValueError('Unexpected range response')
                try:
                    b = r.read()
                except http.client.IncompleteRead as partial:
                    b = partial.partial
                if 0 < len(b) < length:
                    b += fetch(disk, start+len(b), length-len(b))
                assert len(b) == length
                return b
        except Exception as error:
            print(json.dumps({'retry':attempt,'disk':disk,'start':start,'length':length,'error':str(error)}),flush=True)
            if attempt == 4:
                raise
            time.sleep(60 if getattr(error,'code',None)==429 else min(5 * (attempt+1), 20))


def directory(dest):
    cache = dest / 'directory.json'
    if cache.exists():
        return json.loads(cache.read_text())
    tail = fetch(25, LAST-65557, 65557)
    pos = tail.rfind(b'PK\x05\x06')
    assert pos >= 0
    e = struct.unpack_from('<4s4H2LH', tail, pos)
    disk, n, size, offset = e[2], e[4], e[5], e[6]
    if n == 65535 or size == 4294967295:
        loc = struct.unpack_from('<4sLQL', tail, pos-20)
        assert loc[0] == b'PK\x06\x07'
        ext = struct.unpack('<4sQ2H2L4Q', fetch(loc[1], loc[2], 56))
        assert ext[0] == b'PK\x06\x06'
        disk, n, size, offset = ext[5], ext[7], ext[8], ext[9]
    # The central directory is 187 MB (1.48 million image/pose entries).
    chunks = [(a, min(4*1024*1024, size-a)) for a in range(0,size,4*1024*1024)]
    def chunk(pair):
        a, length = pair
        path = dest / ('directory-chunks/%09d.bin' % a)
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(fetch(disk, offset+a, length))
        return path.read_bytes()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        b = b''.join(pool.map(chunk, chunks))
    out = []
    p = 0
    for _ in range(n):
        h = struct.unpack_from('<4s6H3L5H2L', b, p)
        assert h[0] == b'PK\x01\x02' and not h[3] & 1
        name = b[p+46:p+46+h[10]].decode('utf8')
        if name.lower().endswith(('.json','.csv','.txt')):
            out.append(dict(name=name, method=h[4], crc=h[7], compressed=h[8], size=h[9], disk=h[13], offset=h[16]))
        p += 46+h[10]+h[11]+h[12]
    assert p == len(b)
    save(cache, out)
    save(dest/'directory-audit.json', {'total_entries':n,'selected_entries':len(out),'central_directory_bytes':size})
    return out


def get_members(entries, dest):
    # Central-directory order is not assumed to match physical order.
    selected = sorted(entries, key=lambda x: (x['disk'], x['offset']))
    groups = []
    for e in selected:
        end = e['offset'] + 30 + len(e['name'].encode()) + 4096 + e['compressed']
        if groups and groups[-1]['disk'] == e['disk'] and e['offset'] - groups[-1]['end'] < 32768 and end - groups[-1]['start'] < 8*1024*1024:
            groups[-1]['entries'].append(e)
            groups[-1]['end'] = max(end, groups[-1]['end'])
        else:
            groups.append(dict(disk=e['disk'], start=e['offset'], end=end, entries=[e]))

    def work(g):
        archive=dest/('pose-archives/%02d-%09d.zip'%(g['disk'],g['start']))
        proof=archive.with_suffix('.json')
        if ARCHIVE_MEMBERS and archive.exists() and proof.exists():return 0
        pending = g['entries'] if ARCHIVE_MEMBERS else [e for e in g['entries'] if not (dest / e['name']).exists()]
        if not pending:
            return 0
        raw = fetch(g['disk'], g['start'], g['end']-g['start'])
        memory=io.BytesIO()
        pack=zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) if ARCHIVE_MEMBERS else None
        hashes=[]
        for e in pending:
            p = e['offset']-g['start']
            h = struct.unpack_from('<4s5H3L2H', raw, p)
            assert h[0] == b'PK\x03\x04'
            assert raw[p+30:p+30+h[9]].decode() == e['name']
            a = p+30+h[9]+h[10]
            packed = raw[a:a+e['compressed']]
            b = zlib.decompress(packed, -15) if e['method'] == 8 else packed
            assert len(b) == e['size'] and zlib.crc32(b) & 0xffffffff == e['crc']
            if pack is not None:
                pack.writestr(e['name'],b)
                hashes.append({'path':e['name'],'crc32':e['crc'],'sha256':hashlib.sha256(b).hexdigest()})
                continue
            path = (dest/e['name']).resolve()
            assert dest.resolve() in path.parents, 'Unsafe archive path'
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix+'.part')
            tmp.write_bytes(b)
            tmp.replace(path)
        if pack is not None:
            pack.close();archive.parent.mkdir(exist_ok=True)
            tmp=archive.with_suffix('.part');tmp.write_bytes(memory.getvalue());tmp.replace(archive)
            save(proof,{'archive':archive.name,'archive_sha256':hashlib.sha256(memory.getvalue()).hexdigest(),'files':hashes})
        return len(pending)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for count in pool.map(work, groups):
            done += count
            print(json.dumps({'downloaded_members': done, 'selected_members':len(entries)}), flush=True)
    return {'selected_members':len(entries), 'range_groups':len(groups), 'requested_bytes':sum(g['end']-g['start'] for g in groups)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--mode', choices=['index','metadata','poses'], default='index')
    a = ap.parse_args()
    dest = a.dest.resolve()
    assert str(dest).lower().startswith('d:\\gait-research-data\\d_p\\') or str(dest).startswith('/abr/RESEARCH_USER/'), 'Large data must use approved storage'
    dest.mkdir(parents=True, exist_ok=True)
    entries = directory(dest)
    poses = [e for e in entries if e['name'].lower().endswith('.json')]
    csvs = [e for e in entries if e['name'].lower().endswith(('.csv','.txt'))]
    if a.mode == 'index':
        print(json.dumps({'entries':len(entries),'pose_files':len(poses),'pose_compressed_bytes':sum(e['compressed'] for e in poses),'pose_examples':[e['name'] for e in poses[:8]],'metadata':csvs}), flush=True)
        return
    selected = csvs if a.mode == 'metadata' else poses
    receipt = get_members(selected, dest)
    if ARCHIVE_MEMBERS:
        receipt['archives']=[json.loads(p.read_text()) for p in sorted((dest/'pose-archives').glob('*.json'))]
        receipt['files']=[e for p in receipt['archives'] for e in p['files']]
        assert len(receipt['files'])==len(selected)
    else:
        receipt['files'] = [{'path':e['name'],'crc32':e['crc'],'sha256':hashlib.sha256((dest/e['name']).read_bytes()).hexdigest()} for e in selected]
    save(dest / ('receipt-'+a.mode+'.json'), receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','archives')}), flush=True)


if __name__ == '__main__':
    main()
