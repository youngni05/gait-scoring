"""Download pinned public GAVD CSV annotations to D and audit frozen-CNN candidates.

No pickle loading, video download, pose estimation, model inference or time resampling.
"""
import argparse
import ast
import collections
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
DATA = Path('D:/gait-research-data/D_P/datasets/GAVD')
API = 'https://api.github.com/repos/Rahmyyy/GAVD'


def get(url, timeout=45):
    req = urllib.request.Request(url, headers={'User-Agent': 'GAVD-research-audit/1.0'})
    return urllib.request.urlopen(req, timeout=timeout)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing an empty table without a schema')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = HERE / 'source_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    else:
        with get(API + '/commits/main') as r:
            commit = json.load(r)['sha']
        with get(API + '/contents/data?ref=' + commit) as r:
            files = [x for x in json.load(r) if x['name'].endswith('.csv')]
        manifest = dict(commit=commit, repository='https://github.com/Rahmyyy/GAVD',
                        retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        files=[dict(name=x['name'], size=x['size'], git_blob_sha=x['sha'],
                                    url=x['download_url']) for x in files])
        write_json(manifest_path, manifest)
    for info in manifest['files']:
        path = DATA / 'annotations' / info['name']
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            temp = path.with_suffix('.csv.partial')
            with get(info['url'], 120) as response, temp.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            os.replace(temp, path)
        content = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if len(content) != info['size'] or blob != info['git_blob_sha']:
            raise ValueError('Source integrity mismatch: ' + str(path))
        info.update(local_path=str(path), sha256=hashlib.sha256(content).hexdigest())
        print('verified', path.name, len(content), flush=True)
    write_json(manifest_path, manifest)


def contiguous_run(frames):
    best = run = 0
    last = None
    for frame in sorted(frames):
        run = run + 1 if last is not None and frame == last + 1 else 1
        best = max(best, run)
        last = frame
    return best


def audit():
    manifest = json.loads((HERE / 'source_manifest.json').read_text(encoding='utf-8'))
    seqs = {}
    count = 0
    for info in manifest['files']:
        path = Path(info['local_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('Modified annotations: ' + str(path))
        with path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                count += 1
                seq = row['seq']
                s = seqs.setdefault(seq, dict(frames=set(), duplicates=0,
                     meta={k:set() for k in ['id','url','dataset','gait_pat','cam_view','vid_info']}))
                frame = int(row['frame_num'])
                if frame in s['frames']:
                    s['duplicates'] += 1
                s['frames'].add(frame)
                for k in s['meta']:
                    s['meta'][k].add(row[k])
    rows = []
    for seq, s in sorted(seqs.items()):
        m = s['meta']
        conflicts = [k for k,v in m.items() if len(v) != 1]
        values = {k:next(iter(v)) if len(v)==1 else json.dumps(sorted(v)) for k,v in m.items()}
        video = ast.literal_eval(values['vid_info']) if 'vid_info' not in conflicts else {}
        longest = contiguous_run(s['frames'])
        rows.append(dict(seq=seq, video_id=values['id'], url=values['url'],
                         label=values['dataset'], gait_pattern=values['gait_pat'], view=values['cam_view'],
                         label_consistent=int((values['dataset']=='Normal Gait') == (values['gait_pat']=='normal')),
                         annotated_frames=len(s['frames']), frame_first=min(s['frames']),
                         frame_last=max(s['frames']), longest_contiguous_frames=longest,
                         duplicate_rows=s['duplicates'], metadata_conflicts=';'.join(conflicts),
                         annotation_video_info=json.dumps(video, ensure_ascii=False),
                         length_candidate=int(longest >= 124 and not conflicts and not s['duplicates']),
                         pose_quality_status='not_extracted', actual_input_ready=0))
    write_csv(HERE / 'sequences.csv', rows)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['label'], row['view'])].append(row)
    grouped = []
    for (label, view), items in sorted(groups.items()):
        eligible = [r for r in items if r['length_candidate']]
        grouped.append(dict(label=label, view=view, sequences=len(items),
                            unique_videos=len({r['video_id'] for r in items}),
                            length_candidate_sequences=len(eligible),
                            length_candidate_videos=len({r['video_id'] for r in eligible})))
    write_csv(HERE / 'counts_by_label_view.csv', grouped)
    candidates = [r for r in rows if r['length_candidate']]
    if candidates:
        write_csv(HERE / 'length_candidates.csv', candidates)
    summary = dict(source_commit=manifest['commit'], annotation_rows=count, sequences=len(rows),
                   unique_videos=len({r['video_id'] for r in rows}),
                   labels=dict(collections.Counter(r['label'] for r in rows)),
                   views=dict(collections.Counter(r['view'] for r in rows)),
                   patterns=dict(collections.Counter(r['gait_pattern'] for r in rows)),
                   length_candidates=dict(collections.Counter(r['label'] for r in candidates)),
                   duplicate_rows=sum(s['duplicates'] for s in seqs.values()),
                   metadata_conflict_sequences=sum(bool(r['metadata_conflicts']) for r in rows),
                   inconsistent_label_sequences=sum(not r['label_consistent'] for r in rows),
                   actual_input_ready=0, grouped=grouped)
    write_json(HERE / 'annotation_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def probe_one(row):
    url = 'https://www.youtube.com/oembed?' + urllib.parse.urlencode({'url':row['url'], 'format':'json'})
    result = dict(video_id=row['video_id'], url=row['url'],
                  checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  status='', http_status='', title='', error='')
    try:
        with get(url, timeout=20) as r:
            data = json.load(r)
            result.update(status='oembed_available', http_status=r.status, title=data.get('title',''))
    except urllib.error.HTTPError as e:
        result.update(status='oembed_http_error', http_status=e.code, error=str(e))
    except (OSError, ValueError) as e:
        result.update(status='probe_error', error=str(e))
    return result


def probe(limit):
    with (HERE / 'length_candidates.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    videos = {r['video_id']:r for r in rows}
    cache_path = DATA / 'video_access_oembed.json'
    cached = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    pending = [r for k,r in sorted(videos.items()) if k not in cached]
    if limit:
        pending = pending[:limit]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(probe_one, pending):
            cached[result['video_id']] = result
            write_json(cache_path, cached)
            if len(cached) % 20 == 0:
                print('probed', len(cached), '/', len(videos), flush=True)
    results = [cached[k] for k in sorted(videos) if k in cached]
    if results:
        write_csv(HERE / 'video_access.csv', results)
    summary = dict(candidate_unique_videos=len(videos), checked_videos=len(results),
                   statuses=dict(collections.Counter(r['status'] for r in results)),
                   http_statuses=dict(collections.Counter(str(r['http_status']) for r in results)),
                   limitation='oEmbed availability is NOT video download, frame correspondence, pose quality, or input readiness')
    write_json(HERE / 'access_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['download','audit','probe'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'download':
        download()
    elif args.command == 'audit':
        audit()
    else:
        probe(args.limit)
