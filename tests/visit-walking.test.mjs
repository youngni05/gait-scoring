import assert from 'node:assert/strict';
import test from 'node:test';
import { directionHistory, walkingEstimate } from '../mobile/src/ui/data/visit-walking.mjs';

function visit(id, date, directions) {
  const reports = [];
  const segments = directions.map((direction, i) => {
    const start_ms = i * 4000;
    const tracking = {status:'no_signal_in_checked_transitions', checked_transitions:99,
      possible_transitions:99, transitions:[], contact_radius_ms:75};
    reports.push({segment_id:String(i), metrics:[{id:'rhythm', evidence:{tracking}}]});
    const steps = ['left','right','left'].map((landing_side, j) => ({
      landing_side, start_ms:start_ms+j*500, stop_ms:start_ms+(j+1)*500
    }));
    return {segment_id:String(i), direction, start_ms, stop_ms:start_ms+3000, gait:{quality_reasons:[], steps}};
  });
  return {id, subject_id:'s', captured_at:date, capture_mode:'guided', tracking_policy_id:'gpu',
    source_sha256:id, analysis:{source_id:id, current_report:{segments:reports}, segments}};
}

test('use actual intervals without spanning passes, remove only intervals near a switch', () => {
  const v = visit('a','2026-01-03',['image_left','image_right']);
  v.analysis.current_report.segments[0].metrics[0].evidence.tracking.transitions.push({source_time_ms:970});
  const r = walkingEstimate(v);
  assert.deepEqual([r.count,r.cadence,r.left.count,r.right.count],[4,120,3,1]);
  assert.equal(walkingEstimate(v,'image_left').cadence,null);
  assert.equal(walkingEstimate(v,'image_left').left.value,.5);
});

test('a rejected pass requires locally reprocessed intervals; old records can stay unavailable', () => {
  const v = visit('a','2026-01-03',['image_left']);
  v.analysis.segments[0].gait.quality_reasons.push('pose_absent');
  assert.equal(walkingEstimate(v).count,0);
  v.local_step_intervals_by_segment = {'0':v.analysis.segments[0].gait.steps.slice(1)};
  assert.equal(walkingEstimate(v).count,2);
});

test('group by travel direction across different segment counts, capture modes and extractors', () => {
  const current = visit('c','2026-01-03',['image_left','image_left','image_right']);
  const previous = visit('b','2026-01-02',['image_left','image_right','image_right']);
  previous.tracking_policy_id = 'cpu';
  previous.capture_mode = 'fixed';
  const duplicate = {...previous,id:'duplicate',source_sha256:'c'};
  const unrelated = {...previous,id:'other',subject_id:'other'};
  const groups = directionHistory(current,[duplicate,unrelated,previous]).groups;
  assert.deepEqual(groups.map(g => [g.direction,g.previousId,g.currentSegments,g.previousSegments,g.rows[0].currentCount,g.rows[0].previousCount,g.notes.length]),
    [['image_left','b',2,1,6,3,2],['image_right','b',1,2,3,6,2]]);
  assert.equal(directionHistory(current,[duplicate]).groups.length,0);
  assert.equal(directionHistory(previous,[current]).groups.length,0);
});
