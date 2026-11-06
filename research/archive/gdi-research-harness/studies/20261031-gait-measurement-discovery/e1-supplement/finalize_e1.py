"""Final source, report-number and artifact checks; visual QA is recorded separately."""
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_e1 as run
import pandas as pd
import sklearn


def main():
    frozen = run.check_frozen(all_sources=True)
    p = run.protocol()
    names = run.names()
    manifest = run.read(HERE / 'report-manifest.json')
    assert manifest['storyline_only_slide8_replaced']
    for path, digest in manifest['sha256'].items():
        assert run.parent.sha(path) == digest, path
    visual = run.read(HERE / 'visual-review.json')
    assert visual['passed'] and visual['image_sha256'] == run.parent.sha(HERE / 'slide8-e1.png')
    turn = HERE.parent.parent / '20261031-opencap-turn/STATUS.md'
    assert manifest['turn_reference_source_sha256'] == run.parent.sha(turn)
    results = (HERE / 'RESULTS.md').read_text(encoding='utf-8')
    slide = (HERE / 'slide8.md').read_text(encoding='utf-8')
    story_path = HERE.parents[3] / 'docs/interim-storyline-v2.md'
    story = story_path.read_text(encoding='utf-8')
    section = story[story.index('### 8.'):story.index('\n---\n\n## 3막.', story.index('### 8.'))]
    assert slide.rstrip() == section.replace('../gdi-research-harness/studies/20261031-gait-measurement-discovery/e1-supplement/', '').rstrip()
    all_groups = pd.read_csv(HERE / 'all-groups.csv')
    frames, checks = [], []
    fits = 0
    for name in names:
        verified = run.read(HERE / f'{name}-verification.json')
        assert verified['passed'] and verified['protocol_sha256'] == frozen['protocol_sha256']
        assert verified['verifier_sha256'] == run.parent.sha(HERE / 'verify_e1.py')
        checks.append(verified)
        execution = run.read(run.OUT / name / 'execution.json')
        assert execution['status'] == 'complete' and execution['protocol_sha256'] == frozen['protocol_sha256']
        for marker_path in sorted((run.OUT / name).glob('r*f*.complete.json')):
            marker = run.read(marker_path)
            assert marker['protocol_sha256'] == frozen['protocol_sha256']
            for path, digest in marker['sha256'].items():
                assert run.parent.sha(path) == digest, path
        fits += execution['control_fits']
        summary = run.read(HERE / f'{name}-summary.json')
        ci = summary['full_minus_speed']
        text = f'{summary["speed_MAE"]:.3f} | {summary["full_MAE"]:.3f} | {ci["delta"]:+.3f} [{ci["ci_low"]:+.3f}, {ci["ci_high"]:+.3f}]'
        assert text in results
        if name in p['primary_datasets']:
            assert text in slide
        group_path = HERE / f'{name}-groups.csv'
        assert summary['group_table_sha256'] == run.parent.sha(group_path)
        frame = pd.read_csv(group_path)
        frames.append(frame)
        def displayed(delta, low, high):
            def number(v):
                return format(v, '+.6f') if v != 0 and abs(v) < .0005 else format(v, '+.3f')
            return number(delta) + ' [' + number(low) + ', ' + number(high) + ']'
        for _, row in frame.iterrows():
            assert displayed(row.removal_delta, row.removal_ci_low, row.removal_ci_high) in results
            if row.family != 'speed':
                assert displayed(row.addition_delta, row.addition_ci_low, row.addition_ci_high) in results
                assert displayed(row.real_minus_shuffle, row.shuffle_ci_low, row.shuffle_ci_high) in results
    pd.testing.assert_frame_equal(all_groups, pd.concat(frames, ignore_index=True))
    assert fits == 1800 and len(all_groups) == 30 and int(all_groups.addition_delta.notna().sum()) == 24
    for document in [HERE / 'RESULTS.md', HERE / 'slide8.md', story_path]:
        text = document.read_text(encoding='utf-8')
        if document == story_path:
            text = section
        for link in re.findall(r'\]\(([^)]+)\)', text):
            assert (document.parent / link).resolve().exists(), (document, link)
    size = sum(path.stat().st_size for path in run.OUT.rglob('*') if path.is_file())
    result = {'passed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_recorded_before_execution': datetime.fromisoformat(frozen['created_utc']).timestamp()
             <= min((run.OUT / name / 'r0f0.complete.json').stat().st_mtime for name in names),
        'frozen_source_hashes_unchanged': len(frozen['sha256']),
        'datasets': checks, 'control_fits': fits, 'outer_cells': sum(c['outer_cells'] for c in checks),
        'control_prediction_rows': sum(c['all_control_rows_checked'] for c in checks),
        'paired_10000_draw_intervals_independently_checked': sum(c['independent_10000_draw_intervals_checked'] for c in checks),
        'all_addition_comparisons': 24, 'all_removal_comparisons': 30,
        'new_fit_and_checkpoint_checks': sum(c['original_RF_fit_replays']+c['shuffled_RF_fit_replays'] for c in checks),
        'all_report_numbers_and_local_links_checked': True, 'only_storyline_slide8_replaced': True,
        'figure_visually_reviewed': True,
        'D_storage_bytes': size, 'small_reports_on_C': True,
        'environment': {'python': sys.version.split()[0], 'executable': sys.executable,
                        'numpy': run.np.__version__, 'pandas': pd.__version__,
                        'scikit_learn': sklearn.__version__, 'device': 'CPU', 'new_packages': False},
        'protocol_sha256': run.parent.sha(HERE / 'protocol.json'),
        'finalizer_sha256': run.parent.sha(Path(__file__)),
        'limits': 'Not all forests refitted, no external clinical/generalization or video measurement validation.'}
    assert result['protocol_recorded_before_execution']
    run.write(HERE / 'final-verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'datasets'}), flush=True)


if __name__ == '__main__':
    main()
