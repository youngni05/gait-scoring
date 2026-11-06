"""Freeze cohorts and identical raw intervals before new association calculations."""
from common import *


def main():
    assert not (HERE / 'input.npz').exists(), 'Preserve completed preparation'
    protocol = read(HERE / 'protocol.json')
    old = arrays(TRANSFER / 'prepared.npz')
    quality = read(source(TRANSFER / 'quality-final.json'))
    assert quality['reviewed_all_tracks'] and quality['before_cohort_predictions']
    tracks = {r['person']: r for r in read(source(TRANSFER / 'track-review.json'))}
    normalize = functions(ROOT / 'tools/stanford-sample-check/prepare_ridge_data.py', ['normalize'])['normalize']
    converter = functions(MB / 'assets/mmpose_inference_3d.py', ['convert_keypoint_definition'])['convert_keypoint_definition']
    mapping = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
    sys.path.insert(0, str(BUNDLE))
    from preprocess import extract_windows
    for name in ['preprocess.py', 'sources/video_process_utils.py', 'sources/process_raw_videos.ipynb']:
        source(BUNDLE / name)
    kept, converted, audit = [], [], []
    for i, pid in enumerate(old['ids']):
        pid = str(pid)
        p = source(TRANSFER / 'tracks' / tracks[pid]['track_file'], quality['track_sha256'][pid])
        assert sha(p) == tracks[pid]['sha256'] and pid not in quality['exclusions']
        raw = arrays(p)['raw']
        rebuilt = extract_windows(raw)['x']
        np.testing.assert_array_equal(rebuilt[0], old['x'][i])
        body = raw.reshape(124, 25, 3).astype(np.float32)
        assert np.isfinite(body).all() and (body[:, [9, 10, 11, 12, 13, 14], 2] > 0).all()
        h = converter(body[:, mapping], 'coco', 'h36m')
        x = normalize(h)
        valid = np.array_equal(x[..., 2], h[..., 2])
        audit.append({'person': pid, 'h36m_accepted': valid,
                      'confidence_above_one': int((h[..., 2] > 1).sum()),
                      'reason': None if valid else 'normalization_would_change_confidence'})
        if valid:
            kept.append(i)
            converted.append(x)
    ids = old['ids'][kept]
    assert ids.tolist() == protocol['primary_people']
    labels = clinical()
    y = np.array([labels[str(p)] for p in ids])
    np.testing.assert_array_equal(y[:, 0], old['y'][kept])
    original = arrays(TORONTO / 'inputs.npz')
    supplemental_ids = np.array([str(int(p)) for p in original['person']])
    np.savez_compressed(HERE / 'input.npz', ids=ids, labels=y,
        x17=np.stack(converted), x12=old['x'][kept],
        h36m_x17=original['input'], h36m_owner=supplemental_ids,
        cnn_x12=old['x'], cnn_ids=old['ids'], cnn_labels=np.array([labels[str(p)] for p in old['ids']]))
    save('input-audit.json', {'primary_people': ids.tolist(), 'primary_records': len(ids),
        'h36m_supplement_people': len(set(supplemental_ids)), 'h36m_supplement_records': len(supplemental_ids),
        'cnn_supplement_people': len(old['ids']), 'cohort_audit': audit,
        'existing_tracking_exclusions': quality['exclusions'], 'preprocessing_replays_exact': True,
        'fps_policy': 'Primary/CNN native124 frames; existing H36M supplement uses previous 30Hz interpolation.',
        'protocol_sha256': sha(HERE / 'protocol.json'), 'input_sha256': sha(HERE / 'input.npz')})
    save('preparation-sources.json', SOURCES)
    print(json.dumps(read(HERE / 'input-audit.json'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
