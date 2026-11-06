"""Check public video metadata without credentials, video download, or bypasses."""
import argparse
import collections
import concurrent.futures
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from audit_gavd import HERE, DATA, write_csv, write_json


def probe(row):
    folder = DATA / 'stream_metadata'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / (row['video_id'] + '.json')
    error = folder / (row['video_id'] + '.log')
    result = dict(video_id=row['video_id'], status='', formats=0, duration=None,
                  fps_options=[], checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    if output.exists():
        data = json.loads(output.read_text(encoding='utf-8'))
    else:
        cmd = [sys.executable, '-m', 'yt_dlp', '--skip-download', '--dump-single-json',
               '--no-playlist', '--no-cache-dir', '--no-warnings', '--socket-timeout', '12',
               '--retries', '0', '--extractor-retries', '0', '--', row['url']]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=50)
            error.write_text(p.stderr, encoding='utf-8')
            if p.returncode:
                return dict(result, status='metadata_unavailable', error=p.stderr[-1200:])
            data = json.loads(p.stdout)
            write_json(output, data)
        except (subprocess.TimeoutExpired, ValueError, OSError) as e:
            return dict(result, status='probe_error', error=str(e)[-1200:])
    formats = [f for f in data.get('formats',[]) if f.get('vcodec') not in (None,'none') and f.get('url')]
    return dict(result, status='video_formats_listed' if formats else 'no_video_formats',
                formats=len(formats), duration=data.get('duration'),
                fps_options=sorted({f['fps'] for f in formats if f.get('fps')}), error='')


def main(limit):
    with (HERE/'length_candidates.csv').open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.DictReader(f))
    with (HERE/'video_access.csv').open(encoding='utf-8-sig',newline='') as f:
        accessible = {r['video_id'] for r in csv.DictReader(f) if r['status']=='oembed_available'}
    # Check both labels first; stable order, no score-dependent selection.
    videos = {}
    for label in ['Normal Gait','Abnormal Gait']:
        for r in rows:
            if r['label']==label and r['video_id'] in accessible:
                videos.setdefault(r['video_id'],r)
    normal = [r for r in videos.values() if r['label']=='Normal Gait']
    abnormal = [r for r in videos.values() if r['label']=='Abnormal Gait']
    pending = []
    for i in range(max(len(normal),len(abnormal))):
        for group in (normal,abnormal):
            if i < len(group): pending.append(group[i])
    path = DATA/'stream_probe_results.json'
    cache = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    pending = [r for r in pending if r['video_id'] not in cache]
    if limit: pending = pending[:limit]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(probe,pending):
            cache[result['video_id']] = result
            write_json(path,cache)
            print(result['video_id'],result['status'],flush=True)
    write_json(HERE/'stream_probe_summary.json',dict(checked=len(cache),
        statuses=dict(collections.Counter(x['status'] for x in cache.values())),
        limitation='Metadata enumeration only; no video decoded, frame mapping or pose quality verified'))
    write_csv(HERE/'stream_probe.csv',list(cache.values()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=6)
    main(parser.parse_args().limit)
