import assert from 'node:assert/strict';
import test from 'node:test';
import { compareVisits, describeVisit, directionGdi, stepGap, visitGdi, visitHistory, visitSummary }
  from '../mobile/src/ui/data/visit-summary.mjs';

// 직접 만든 가상 방문 기록. 실제 측정값이 아니라 계산 규칙을 확인하기 위한 값이다.
function seg(id, direction, start, stop, gdi, opts = {}) {
  const steps = (opts.steps ?? ['left', 'right', 'left', 'right']).map((landing_side, j) => ({
    landing_side, start_ms: start + j * (opts.step ?? 500) + (landing_side === 'left' ? 0 : 0),
    stop_ms: start + (j + 1) * (opts.step ?? 500) + (landing_side === 'left' ? (opts.leftExtra ?? 0) : 0),
  }));
  const tracking = { status: 'ok', checked_transitions: 1, possible_transitions: 1, transitions: [], contact_radius_ms: 75 };
  const side = n => ({ valid_frames: n, first_source_ms: start, last_source_ms: start + (n - 1) * 33 });
  const frames = opts.frames ?? 60;
  return {
    segment: { segment_id: id, direction, start_ms: start, stop_ms: stop, gait: { quality_reasons: [], steps },
      gdi: gdi ? { models: { normal100: { status: 'available', value: { left: gdi[0], right: gdi[1] } } } } : { models: {} } },
    report: { segment_id: id, metrics: [
      { id: 'rhythm', evidence: { tracking } },
      { id: 'knee', evidence: { source_frames_contiguous: true, source_frame_count: frames,
        coverage: { left: side(frames), right: side(frames) }, supported_range_deg: { left: opts.knee?.[0] ?? 55, right: opts.knee?.[1] ?? 55 } } },
    ] },
  };
}
function visit(id, date, parts) {
  return { id, subject_id: 's', captured_at: date + 'T00:00:00+09:00', capture_mode: 'guided', tracking_policy_id: 'p',
    source_sha256: id, analysis: { source_id: id, provenance: { extractor: { name: 'MediaPipe', delegate: 'GPU' } },
      segments: parts.map(p => p.segment), current_report: { segments: parts.map(p => p.report) } } };
}

test('방향 안에서는 구간 길이 비중, 겹치는 구간은 먼저 나온 것만 사용', () => {
  const v = visit('a', '2026-11-01', [
    seg('long', 'image_left', 0, 6000, [80, 70]),
    seg('short', 'image_left', 6000, 8000, [90, 90]),
    seg('overlap', 'image_left', 5000, 7000, [0, 0]),
  ]);
  const d = directionGdi(v, 'image_left');
  assert.equal(d.left, 82.5);
  assert.equal(d.right, 75);
  assert.deepEqual(d.segments.map(s => [s.segmentId, s.share]), [['long', 0.75], ['short', 0.25]]);
});

test('두 방향은 같은 비중, 마지막에 좌우 평균으로 하나의 GDI', () => {
  const v = visit('a', '2026-11-01', [
    seg('l1', 'image_left', 0, 6000, [80, 70]), seg('l2', 'image_left', 6000, 8000, [90, 90]),
    seg('r1', 'image_right', 9000, 12000, [100, 100]),
  ]);
  const g = visitGdi(v);
  assert.deepEqual([g.left, g.right, g.value], [91.25, 87.5, 89.375]);
  const one = visitGdi(v, ['image_right']);
  assert.deepEqual([one.value, one.directions], [100, ['image_right']]);
  assert.equal(visitGdi(visit('b', '2026-11-01', [seg('x', 'image_left', 0, 3000, null)])).value, null);
});

test('좌우 걸음 시간 차이는 양발이 함께 관측된 같은 구간 안에서만 만든다', () => {
  const v = visit('a', '2026-11-01', [
    seg('onlyLeft', 'image_left', 0, 3000, [90, 90], { steps: ['left', 'left'] }),
    seg('onlyRight', 'image_left', 4000, 7000, [90, 90], { steps: ['right', 'right'] }),
  ]);
  assert.equal(stepGap(visitSummary(v).walking).value, null);
  const both = visit('b', '2026-11-01', [seg('both', 'image_left', 0, 3000, [90, 90], { leftExtra: 100 })]);
  assert.ok(Math.abs(stepGap(visitSummary(both).walking).value - 0.1) < 1e-9);
});

test('지난 기록과는 공통 방향만으로 다시 계산해 비교하고, 예시 기준 안이면 비슷함', () => {
  const prior = visit('p', '2026-11-01', [seg('l', 'image_left', 0, 4000, [80, 80], { knee: [50, 55] })]);
  const current = visit('c', '2026-11-08', [
    seg('l', 'image_left', 0, 4000, [86, 85], { knee: [44, 58] }),
    seg('r', 'image_right', 5000, 9000, [40, 40]),
  ]);
  const cmp = compareVisits(current, prior);
  assert.deepEqual(cmp.directions, ['image_left']);
  assert.deepEqual(cmp.skipped, ['image_right']);
  const row = key => cmp.rows.find(r => r.key === key);
  assert.equal(row('gdi').delta, 5.5);           // 공통 방향만: 85.5 − 80, 오른쪽 방향 40점은 섞이지 않음
  assert.equal(row('gdi').verdict, 'better');
  assert.equal(row('kneeLeft').verdict, 'worse'); // −6°
  assert.equal(row('kneeRight').verdict, 'similar'); // +3°
  assert.equal(compareVisits(current, visit('q', '2026-11-01', [seg('r', 'image_right', 0, 4000, [80, 80])])).directions[0], 'image_right');
});

test('최근 비교를 문장으로 해설하고, 세 번 이상 같은 방향으로 움직이면 추이로 알린다', () => {
  const at = (id, date, g) => visit(id, date, [seg('l', 'image_left', 0, 4000, [g, g])]);
  const h = visitHistory(at('c', '2026-11-15', 90), [at('b', '2026-11-08', 84), at('a', '2026-11-01', 78)]);
  assert.equal(h.recent.previousId, 'b');
  assert.match(h.lines[0], /^지난 기록\(2026\.11\.08\)과 비교하면 GDI\(\+6\.0점\)가 좋아졌습니다\.$/);
  assert.match(h.lines.join(' '), /분당 걸음 수, 좌우 걸음 시간 차이, 왼쪽 무릎 움직임 폭, 오른쪽 무릎 움직임 폭은 비슷합니다/);
  assert.match(h.lines.join(' '), /GDI는 최근 3번의 기록에서 계속 늘었습니다/);
  const gdi = h.trends.find(t => t.key === 'gdi');
  assert.deepEqual(gdi.points.map(p => p.value), [78, 84, 90]);
  assert.deepEqual([gdi.since.delta, gdi.since.verdict], [12, 'better']);
  assert.equal(visitHistory(at('c', '2026-11-15', 90), []).recent, null);
});

test('기록 하나의 해설은 GDI 정의와 좌우 차이만, 항목·짧은 결론으로', () => {
  const v = visit('a', '2026-11-01', [seg('l', 'image_left', 0, 4000, [88, 82], { knee: [48, 56] })]);
  const items = describeVisit(visitSummary(v)).map(i => [i.label, i.text, i.notice]);
  assert.deepEqual(items, [
    ['GDI', '정상 평균보다 15.0점 낮음 (약 1.5표준편차)', true],
    ['GDI 좌우', '오른쪽 다리가 6.0점 낮음', true],
    ['걸음 시간', '좌우 차이 없음 (측정 오차 범위)', false],
    ['무릎 움직임 폭', '왼쪽이 8.0° 작음', true],
  ]);
});

test('해설 간소화: GDI가 1표준편차 안이면 생략, 0.07초 차이는 차이 없음', () => {
  const v = visit('a', '2026-11-01', [
    seg('l', 'image_left', 0, 4000, [94, 94], { leftExtra: 70, knee: [59, 30] }),
    seg('r', 'image_right', 5000, 9000, [94, 94], { leftExtra: 70, knee: [30, 56] }),
  ]);
  v.analysis.current_report.segments[0].metrics[1].evidence.coverage.right.valid_frames = 1;
  v.analysis.current_report.segments[1].metrics[1].evidence.coverage.left.valid_frames = 1;
  assert.deepEqual(describeVisit(visitSummary(v)).map(i => i.label + ': ' + i.text),
    ['걸음 시간: 좌우 차이 없음 (측정 오차 범위)', '무릎 움직임 폭: 좌우 비슷함']);
});
