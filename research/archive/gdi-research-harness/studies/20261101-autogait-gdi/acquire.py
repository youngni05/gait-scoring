"""Download official public files; never persist or print signed download credentials."""
from common import *
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import re
import shutil
import urllib.request

PUBLIC = 'https://rochester.app.box.com/v/AtaxiaDataset'


def metadata(file_id):
    request = urllib.request.Request(PUBLIC + '/file/' + str(file_id), headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode()
    match = re.search(r'Box\.prefetchedData\s*=\s*', html)
    assert match, 'Public file metadata missing'
    return json.JSONDecoder().raw_decode(html[match.end():])[0]['preview_metadata']


def download(item):
    path = DATA / 'videos' / item['name']
    receipt = DATA / 'acquisition' / (item['name'] + '.json')
    if receipt.exists():
        record = read(receipt)
        assert record['bytes'] == path.stat().st_size and record['sha256'] == sha(path)
        return record
    meta = metadata(item['id'])
    assert meta.get('permissions', {}).get('can_download') and meta.get('is_download_available')
    url = meta['authenticated_download_url']
    temporary = path.with_suffix('.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': PUBLIC})
    count = 0
    digest, sha1 = hashlib.sha256(), hashlib.sha1()
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open('wb') as stream:
            assert response.status == 200
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                sha1.update(chunk)
                count += len(chunk)
    except Exception as exc:
        # Signed URLs must not enter logs via HTTP exception strings.
        raise RuntimeError('Public download failed for ' + item['name'] + ' (' + type(exc).__name__ + ')') from None
    assert count == item['itemSize'], (item['name'], count, item['itemSize'])
    upstream_sha1 = meta.get('sha1')
    if upstream_sha1:
        assert sha1.hexdigest() == upstream_sha1
    temporary.replace(path)
    record = {'name': item['name'], 'file_id': item['id'], 'path': str(path), 'bytes': count,
              'sha256': digest.hexdigest(), 'sha1': sha1.hexdigest(),
              'provider_sha1_checked': bool(upstream_sha1), 'source': PUBLIC + '/file/' + str(item['id'])}
    save(receipt, record)
    print('acquired', item['name'], count, flush=True)
    return record


def cohort():
    files = read(OLD / 'ataxia-public-files.json')
    videos = {r['name']: r for r in files if r['name'].endswith('.mp4')}
    with (OLD / 'ataxia-ratings.csv').open(newline='') as stream:
        scores = {r['destination_name']: int(r['Score']) for r in csv.DictReader(stream)}
    with (OLD / 'ataxia-mapping.csv').open(newline='') as stream:
        mapping = list(csv.DictReader(stream))
    rows, groups = [], {}
    for row in mapping:
        name = row['destination_name']
        if name not in videos:
            continue
        assert int(row['Score']) == scores[name]
        match = re.match(r'^([A-Z]+)[-_](\d+)[-_](\d+)', row['ID'])
        assert match
        person = '-'.join(match.groups())
        token = re.split(r'[-_]', row['ID'][match.end():].lstrip('-_'))[0]
        date = None
        for fmt in ['%m%d%Y', '%d%b%Y']:
            try:
                candidate = datetime.strptime(token, fmt).date()
                if 2018 <= candidate.year <= 2022:
                    date = candidate.isoformat()
                    break
            except ValueError:
                pass
        visit_match = re.search(r'(?:[-_])Y(\d+)', row['ID'])
        month_match = re.search(r'(24MO|36M|12mo)', row['ID'])
        visit = int(visit_match[1]) if visit_match else int(re.match(r'\d+', month_match[0])[0]) // 12 if month_match else None
        entry = {'video': name, 'person': person, 'date': date, 'source_id': row['ID'],
                 'visit': visit, 'date_token': token, 'SARA': scores[name], 'file_id': videos[name]['id'], 'expected_bytes': videos[name]['itemSize']}
        rows.append(entry)
        groups.setdefault(person, []).append(entry)
    # Filename dates include upstream typos; explicit visit order is authoritative.
    # There must be one earliest-visit video per person, so invalid dates cannot affect selection.
    for items in groups.values():
        if any(r['visit'] is None for r in items):
            assert len(items) == 1, 'Unknown visit in repeated participant'
            items[0]['visit'] = 0
            items[0]['visit_note'] = 'Only public record; visit unspecified'
        earliest = min(r['visit'] for r in items)
        assert sum(r['visit'] == earliest for r in items) == 1
    selected = [min(items, key=lambda r: r['visit']) for items in groups.values()]
    selected.sort(key=lambda r: int(r['video'].split('.')[0]))
    for r in selected:
        r['pre_exclusion'] = 'official_tandem_walking' if r['video'] == '146.mp4' else None
        r['source_note'] = 'official_partial_staff_occlusion' if r['video'] == '63.mp4' else 'shoes' if r['video'] == '144.mp4' else None
    assert len(rows) == 149 and len(selected) == 89
    save(HERE / 'cohort.json', selected)
    save(HERE / 'source-linkage.json', {'linked_videos': len(rows), 'people': len(groups),
         'all_video_score_counts': dict(sorted(Counter(r['SARA'] for r in rows).items())),
         'earliest_visit_score_counts': dict(sorted(Counter(r['SARA'] for r in selected).items())),
         'sha256': {str(OLD / n): sha(OLD / n) for n in ['ataxia-public-files.json', 'ataxia-ratings.csv', 'ataxia-mapping.csv']},
         'selection_precedes_gdi_predictions': True, 'independent_people_not_videos': True})
    return files, selected


def main():
    assert read(HERE / 'protocol.json').get('status') == 'ready', 'Study stopped: viewpoint mismatch; read protocol.json'
    assert DATA.resolve().drive.upper() == 'D:'
    for name in ['videos', 'acquisition']:
        (DATA / name).mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(DATA).free > 55 * 1024 ** 3, 'Insufficient space for bounded pose work'
    files, selected = cohort()
    wanted = {r['video'] for r in selected if not r['pre_exclusion']}
    items = [r for r in files if r['name'] in wanted]
    print(json.dumps({'selected_people': len(selected), 'videos_to_acquire': len(items),
                      'expected_bytes': sum(r['itemSize'] for r in items)}), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, item) for item in items]
        results = [f.result() for f in as_completed(futures)]
    results.sort(key=lambda r: int(r['name'].split('.')[0]))
    save(DATA / 'acquisition.json', results)
    save(HERE / 'acquisition-summary.json', {'videos': len(results), 'bytes': sum(r['bytes'] for r in results),
        'data_directory': str(DATA), 'acquisition_sha256': sha(DATA / 'acquisition.json')})
    print('AUTOGAIT_EARLIEST_VISIT_ACQUISITION_COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    main()
