"""원본 검수 결과를 영상 목록과 사람별 집계로 정리한다. 학습은 실행하지 않는다."""
import collections
import csv
import hashlib
import json
import re
import zlib
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
DATA = Path('D:/gait-research-data/D_P/ljmu-side-audit-20261104')


def write_csv(name, rows):
    with (HERE / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    inventory = json.loads((HERE / 'inventory.json').read_text())
    candidates = [e for e in inventory if re.search(r'_90(?:_ALT)?\.mp4$', e['name'])]
    reviews = json.loads((HERE / 'visual-review.json').read_text())
    for name, expected in [('download-side.json', len(candidates)), ('download-poses.json', 987)]:
        downloads = json.loads((HERE / name).read_text())
        assert len(downloads) == expected and all(x['ok'] for x in downloads), name
    workbook = openpyxl.load_workbook(DATA / 'files/Participant_Anthropometry.xlsx', data_only=True)
    metadata = {r[0]: {'age': r[4], 'sex': r[3]} for r in list(workbook.active.values)[1:]}
    rows = []
    for e in candidates:
        stem = Path(e['name']).stem
        d = json.loads((HERE / 'checks' / (stem + '.json')).read_text())
        review = reviews[stem]
        if stem == 'P8_E2T6_90_ALT':
            assert review['pose_alias_review']['overlay_frames_reviewed'] == [70, 115, 160]
        path = DATA / 'files' / e['name']
        raw = path.read_bytes()
        assert len(raw) == e['size'] and zlib.crc32(raw) & 0xffffffff == e['crc'], stem
        assert d['decode_ok'] and not d.get('error'), stem
        assert d['header_frames'] == d['decoded_frames'], stem
        assert d['monotonic_timestamps'], stem
        assert d['fps_fraction'] == '30/1' and (d['width'], d['height']) == (1280, 720), stem
        reasons = []
        if d['experiment'] not in (1, 2) or review['view'] == 'rear':
            reasons.append('후면 영상: 측면 조건 불충족')
        elif review['view'] != 'side':
            reasons.append('심한 역광으로 측면 및 신체 확인 불충분')
        if d['pose_status'] == 'missing':
            reasons.append('대응 원본 좌표 없음')
        else:
            if not d['video_pose_count_match'] or not d['frame_ids_contiguous']:
                reasons.append('영상과 좌표 프레임 불일치')
            if d['pose_status'] == 'confidence_out_of_range':
                reasons.append('좌표 신뢰도가 기존 허용 범위 0~1 이탈')
            elif d['pose_status'] == 'preprocess_excluded':
                reasons.append('기존 전처리로 유효한 124프레임 구간 생성 불가')
        usable = not reasons
        row = {
            '영상ID': stem, '사람ID': d['person'], '나이': metadata[d['person']]['age'],
            '실험': d['experiment'], '추가옷차림_파일명기준': stem.endswith('_ALT'),
            '측면확인': review['view'] == 'side' and d['experiment'] in (1, 2),
            '현재입력생성가능': usable, '제외사유': ' | '.join(reasons),
            'FPS': d['fps'], '너비': d['width'], '높이': d['height'],
            '영상프레임': d['decoded_frames'], '좌표프레임': d.get('pose_frames', ''),
            '초': d['duration_seconds'], '유효124프레임구간수': len(d.get('baseline_windows', [])) if usable else 0,
            '유효구간시작_0부터': json.dumps(d.get('baseline_windows', [])) if usable else '[]',
            '대표검수구간시작_0부터': d.get('selected_start'),
            '9개기준점연속화면내구간있음': usable and d.get('observed_inframe_fraction') == 1,
            '원본상대경로': e['name'], '원본절대경로': str(path).replace('\\', '/'),
            '좌표압축파일': d.get('pose_file', ''), 'SHA256': hashlib.sha256(raw).hexdigest(),
            '비고': '원본 영상의 ALT 접미사와 같은 폴더 좌표 이름이 다름. 별도 대응 검수.' if stem == 'P8_E2T6_90_ALT' else '',
        }
        rows.append(row)
    rows.sort(key=lambda r: (r['사람ID'], r['영상ID']))
    assert len({r['SHA256'] for r in rows}) == len(rows), '중복 영상 파일을 먼저 검토해야 한다.'
    usable = [r for r in rows if r['현재입력생성가능']]
    side = [r for r in rows if r['측면확인']]
    strict = [r for r in usable if r['9개기준점연속화면내구간있음']]
    def count(rs):
        return {'people': len({r['사람ID'] for r in rs}), 'videos': len(rs),
                'windows_124': sum(r['유효124프레임구간수'] for r in rs)}
    persons = []
    for person in sorted({r['사람ID'] for r in rows}):
        prs = [r for r in rows if r['사람ID'] == person]
        persons.append({'사람ID': person, '나이': metadata[person]['age'],
                        '원본후보영상수': len(prs), '측면확인영상수': sum(r['측면확인'] for r in prs),
                        '현재입력생성가능영상수': sum(r['현재입력생성가능'] for r in prs),
                        '기준점연속화면내영상수': sum(r['9개기준점연속화면내구간있음'] for r in prs)})
    summary = {
        'source_url': 'https://opendata.ljmu.ac.uk/id/eprint/133/',
        'archive_url': 'https://opendata.ljmu.ac.uk/id/eprint/133/3/Gait%20Dataset.zip',
        'source_zip_bytes': 18700445600,
        'archive_total_mp4': sum(e['name'].endswith('.mp4') for e in inventory),
        'candidate_90': count(rows), 'visually_confirmed_side': count(side),
        'current_preprocessing_usable': count(usable), 'continuous_inframe_landmarks_subset': count(strict),
        'exclusions_nonoverlapping_primary_reason': dict(collections.Counter(r['제외사유'].split(' | ')[0] for r in rows if not r['현재입력생성가능'])),
        'exclusions_all_reasons_may_overlap': dict(collections.Counter(reason for r in rows for reason in r['제외사유'].split(' | ') if reason)),
        'usable_experiments': dict(collections.Counter(r['실험'] for r in usable)),
        'age_range': [min(r['나이'] for r in usable), max(r['나이'] for r in usable)],
        'age_65plus_people': len({r['사람ID'] for r in usable if r['나이'] >= 65}),
        'age_groups_people': dict(collections.Counter(f"{(p['나이']//10)*10}s" for p in persons if p['현재입력생성가능영상수'])),
        'source_metadata_rows': len(metadata), 'metadata_id_without_video': sorted(set(metadata)-{r['사람ID'] for r in rows}),
        'video_crc_verified': len(rows), 'all_frame_decode_verified': len(rows),
        'exact_duplicate_video_files': 0, 'pose_sequences_acquired': 987,
        'visual_review': '모든 후보의 대표 프레임 검수 및 의심 사례의 시간순 장면 확인',
        'limitations_ko': [
            '모든 후보의 대표 프레임을 직접 확인했다. 제외 의심 영상은 시간순 장면도 확인했다. 모든 프레임의 자세 정확도를 사람이 판정한 것은 아니다.',
            '현재입력생성가능은 측면 확인, 영상-좌표 프레임 대응, 기존 전처리 통과를 뜻한다. 구간 전체가 일정 속도 보행이거나 좌표가 정확하다는 임상 검증은 아니다.',
            '기준점 연속 화면 내 부분집합은 목, 양쪽 고관절/무릎/발목/엄지발가락의 좌표가 124프레임 모두 화면 경계에서 8픽셀 이상 안쪽에 있는 후보 구간을 포함한다. 추가 참고 조건이며 새로운 임상 품질 기준이 아니다.',
            '유효 구간은 124프레임, 이동 간격 31프레임이며 서로 겹친다. 구간 수를 독립 표본 수로 사용하지 않는다.',
            '한 사람의 반복 촬영과 옷차림 변경 영상은 반드시 같은 학습/검증/평가 분할에 배치한다.',
            'GDI 실측값은 없다. 이 파일들은 GDI=100 정답이나 정상성의 임상 검증을 제공하지 않는다.',
            'LJMU 측면 촬영과 스탠퍼드 추적 회전 촬영은 카메라 조건이 다르다.',
        ],
    }
    write_csv('all-candidates.csv', rows)
    write_csv('usable-videos.csv', usable)
    write_csv('excluded-videos.csv', [r for r in rows if not r['현재입력생성가능']])
    write_csv('people-summary.csv', persons)
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
