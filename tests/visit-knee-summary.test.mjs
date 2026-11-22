import assert from 'node:assert/strict';
import test from 'node:test';
import { visitKneeSummary } from '../mobile/src/ui/data/report-values.mjs';

function pass(id, direction, frames, left, right) {
  const side = count => ({ valid_frames: count, first_source_ms: 100, last_source_ms: 100 + (count - 1) * 40 });
  return {
    segment: { segment_id: id, direction },
    report: { segment_id: id, metrics: [{ id: 'knee', evidence: {
      source_frames_contiguous: true, source_frame_count: frames,
      coverage: { left: side(left), right: side(right) },
      supported_range_deg: { left: 55, right: 61 }
    } }] }
  };
}

function visit(passes, local = {}) {
  return { analysis: {
    segments: passes.map(p => p.segment), current_report: { segments: passes.map(p => p.report) }
  }, local_observations_by_segment: local };
}

const out = pass('out', 'image_left', 100, 100, 2);
const back = pass('back', 'image_right', 80, 1, 80);
const sources = recording => visitKneeSummary(recording).map(r => [r.side, r.segmentId, r.full]);

test('complementary directions preserve anatomical sides and source intervals', () => {
  const rows = visitKneeSummary(visit([out, back]));
  assert.deepEqual(rows.map(r => [r.side, r.segmentId, r.value, r.start, r.stop]), [
    ['left', 'out', 55, 100, 4060], ['right', 'back', 61, 100, 3260]
  ]);
});

test('prefer a complete paired pass, then longer coverage regardless of angle', () => {
  const both = pass('both', 'image_right', 70, 70, 70);
  const short = pass('short', 'image_left', 40, 40, 40);
  short.report.metrics[0].evidence.supported_range_deg.left = 100;
  assert.deepEqual(sources(visit([out, back, short, both])), [['left','both',true], ['right','both',true]]);
});

test('partial ranges use only a sufficiently long continuous local observation', () => {
  const partial = pass('some', 'image_left', 30, 18, 2);
  const row = { segment_id:'some', direction:'image_left', side:'left', unit:'deg',
    status:'limited_local_observation', value:0, valid_count:16, start_ms:100, stop_ms:600 };
  const read = override => visitKneeSummary(visit([partial], { some: [{ ...row, ...override }] }));
  assert.deepEqual(read({}).map(r => [r.side,r.full,r.value]), [['left',false,0]]);
  for (const override of [{ valid_count:2 }, { stop_ms:599 }, { direction:'unknown' }]) {
    assert.deepEqual(read(override), []);
  }
});

test('one visible leg, missing evidence and discontinuous source frames', () => {
  assert.deepEqual(sources(visit([out])), [['left','out',true]]);
  const broken = pass('broken','image_left',30,30,30);
  broken.report.metrics[0].evidence.source_frames_contiguous = false;
  assert.deepEqual(sources(visit([broken])), []);
  assert.deepEqual(sources({ analysis:{ segments:[out.segment] } }), []);
});

test('one-frame gaps can use observed angles; longer internal gaps need a local interval', () => {
  const p = pass('gaps','image_left',30,24,0);
  const curve = Array.from({length:30}, (_,i) => i % 5 === 2 ? null : 45);
  const evidence = p.report.metrics[0].evidence;
  evidence.coverage.left.last_source_ms = 1260;
  evidence.supported_range_deg.left = 0;
  p.segment.knee = { views:{ supported:{ curve_deg:{ left:curve } } } };
  const [row] = visitKneeSummary(visit([p]));
  assert.deepEqual([row.isolatedGaps,row.frames,row.start,row.stop,row.value], [true,24,100,1260,0]);
  curve[7] = 45;
  curve[3] = null;
  assert.deepEqual(sources(visit([p])), []);
});
