const { test } = require('node:test');
const assert = require('node:assert/strict');
const fixture = require('../contracts/coords-1/synthetic.sample.json');
const { validateCoords } = require('../.stage1-build/contracts/coords-1/validate.js');
const copy = () => structuredClone(fixture);

test('synthetic sample: 60 frames, 33 tuples, explicit detection gap', () => {
  assert.equal(fixture.frames.length, 60);
  assert.equal(fixture.frames[30].lm, null);
  for (const frame of fixture.frames) {
    if (frame.lm) {
      assert.equal(frame.lm.length, 33);
      for (const lm of frame.lm) assert.equal(lm.length, 4);
    }
  }
  assert.equal(validateCoords(fixture).ok, true);
});

const invalid = {
  'duplicate timestamp': d => { d.frames[1].t = 0; },
  'decreasing timestamp': d => { d.frames[2].t = 0.01; },
  'negative timestamp': d => { d.frames[0].t = -0.1; },
  'only 59 frames': d => { d.frames.pop(); },
  '32 joints': d => { d.frames[0].lm.pop(); },
  '34 joints': d => { d.frames[0].lm.push([0, 0, 0, 1]); },
  'three components': d => { d.frames[0].lm[0].pop(); },
  'five components': d => { d.frames[0].lm[0].push(1); },
  'missing lm': d => { delete d.frames[0].lm; },
  'missing subject': d => { delete d.subject; },
  'missing mark': d => { delete d.marks.stand_end; },
  'missing timestamp': d => { delete d.frames[0].t; },
  'NaN coordinate': d => { d.frames[0].lm[0][0] = NaN; },
  'Infinity coordinate': d => { d.frames[0].lm[0][2] = Infinity; },
  'NaN time': d => { d.frames[0].t = NaN; },
  'Infinity mark': d => { d.marks.stand_end = Infinity; },
  'NaN height': d => { d.subject.height_cm = NaN; },
  'null component': d => { d.frames[0].lm[0][1] = null; },
  'string component': d => { d.frames[0].lm[0][1] = '0.5'; },
  'low visibility': d => { d.frames[0].lm[0][3] = -0.01; },
  'high visibility': d => { d.frames[0].lm[0][3] = 1.01; },
  'height below range': d => { d.subject.height_cm = 99.9; },
  'height above range': d => { d.subject.height_cm = 220.1; },
  'string height': d => { d.subject.height_cm = '170'; },
  'blank subject id': d => { d.subject.id = ' '; },
  'half-null interval': d => { d.marks.stand_start = null; },
  'reversed interval': d => { d.marks = { stand_start: 1, stand_end: 0.5 }; },
  'zero length interval': d => { d.marks.stand_end = 0; },
  'negative interval': d => { d.marks.stand_start = -1; },
  'interval past coverage': d => { d.marks.stand_end = 50; },
  'invalid UUID': d => { d.recording_id = 'test'; },
  'date without timezone': d => { d.capture.recorded_at = '2026-11-16T10:00:00'; },
  'date with space separator': d => { d.capture.recorded_at = '2026-11-16 10:00:00+09:00'; },
  'portrait on landscape dimensions': d => { d.capture.orientation = 'portrait'; },
  'landscape on portrait dimensions': d => { d.capture.width = 720; d.capture.height = 1280; },
  'nonexistent date': d => { d.capture.recorded_at = '2026-02-30T10:00:00Z'; },
  'nonpositive width': d => { d.capture.width = 0; },
  'fractional height': d => { d.capture.height = 720.5; },
  'invalid model': d => { d.pose.model = 'unconfirmed'; },
  'invalid delegate': d => { d.pose.delegate = 'AUTO'; },
  'unapproved metadata': d => { d.capture.recorded_at_source = 'metadata'; },
  'unapproved detector field': d => { d.marks.detector_version = 'v1'; },
  'video field forbidden by allowlist': d => { d.video = 'local-file'; },
};
for (const [name, mutate] of Object.entries(invalid)) {
  test(`reject ${name}`, () => {
    const d = copy(); mutate(d);
    const result = validateCoords(d);
    assert.equal(result.ok, false);
    assert.ok(result.issues.length > 0);
    assert.ok(result.issues.every(e => e.path && e.message));
  });
}

test('null/non-object payloads return errors', () => {
  for (const d of [null, undefined, [], 'text', 5]) assert.equal(validateCoords(d).ok, false);
});
test('no standing detected is valid, including all-null frames (format only)', () => {
  const d = copy();
  d.marks = { stand_start: null, stand_end: null };
  d.frames.forEach(f => { f.lm = null; });
  assert.equal(validateCoords(d).ok, true);
});
test('off-screen coordinates and relative depth survive unchanged', () => {
  const d = copy(); d.frames[0].lm[0] = [-0.25, 1.25, -3.1234567, 0.8];
  const before = structuredClone(d);
  assert.equal(validateCoords(d).ok, true);
  assert.deepEqual(d, before);
});
test('irregular timestamps retain original precision', () => {
  const d = copy(); d.frames.forEach((f, i) => { f.t = i * i / 1000; });
  const before = structuredClone(d);
  assert.equal(validateCoords(d).ok, true);
  assert.deepEqual(d, before);
});
test('height boundaries inclusive and optional fields supported', () => {
  for (const height of [100, 220]) {
    const d = copy(); d.subject.height_cm = height; d.pose.delegate = 'CPU';
    delete d.capture.device;
    assert.equal(validateCoords(d).ok, true);
  }
});
test('fixed accepted for compatibility; MVP creation guided only', () => {
  const d = copy(); d.capture.mode = 'fixed';
  assert.equal(validateCoords(d).ok, true);
});
test('duration context checks final frame and standing bounds', () => {
  const d = copy(); d.marks.stand_end = 2;
  assert.equal(validateCoords(d).ok, false);
  assert.equal(validateCoords(d, { durationSeconds: 2 }).ok, true);
  for (const durationSeconds of [1, -1, NaN, Infinity]) {
    assert.equal(validateCoords(d, { durationSeconds }).ok, false);
  }
});
test('subject context comparison is separate from authentication', () => {
  assert.equal(validateCoords(fixture, { expectedSubjectId: fixture.subject.id }).ok, true);
  assert.equal(validateCoords(fixture, { expectedSubjectId: 'different' }).ok, false);
});
test('rejected input is not coerced, repaired or stripped', () => {
  const d = copy(); d.subject.height_cm = '170'; d.extra = true;
  const before = structuredClone(d);
  assert.equal(validateCoords(d).ok, false);
  assert.deepEqual(d, before);
});

test('required fields inherited from prototype cannot disappear during JSON serialization', () => {
  for (const key of Object.keys(fixture)) {
    const d = copy(); const value = d[key]; delete d[key];
    Object.setPrototypeOf(d, { [key]: value });
    assert.equal(validateCoords(d).ok, false, key);
  }
  const d = copy();
  delete d.subject.id;
  Object.setPrototypeOf(d.subject, { id: fixture.subject.id });
  assert.equal(validateCoords(d).ok, false);
});
test('portrait and square dimensions remain accepted', () => {
  const d = copy(); d.capture.width = 720; d.capture.height = 1280;
  d.capture.orientation = 'portrait';
  assert.equal(validateCoords(d).ok, true);
  // No square orientation exists in coords-1; do not invent a new enum.
  d.capture.height = 720;
  assert.equal(validateCoords(d).ok, true);
});
test('timezone offsets, UTC and fractional seconds accepted', () => {
  for (const recorded_at of ['2026-11-16T10:00:00Z', '2026-11-16T10:00:00.123456+09:00',
    '2026-11-16T10:00:00-05:30']) {
    const d = copy(); d.capture.recorded_at = recorded_at;
    assert.equal(validateCoords(d).ok, true);
  }
});
test('every required field is enforced, including nested objects', () => {
  for (const key of Object.keys(fixture)) {
    const d = copy(); delete d[key];
    assert.equal(validateCoords(d).ok, false, key);
  }
  for (const [section, keys] of Object.entries({
    subject: ['id', 'height_cm'],
    capture: ['mode', 'recorded_at', 'width', 'height', 'orientation'],
    pose: ['model', 'engine_version'], marks: ['stand_start', 'stand_end'],
  })) for (const key of keys) {
    const d = copy(); delete d[section][key];
    assert.equal(validateCoords(d).ok, false, `${section}.${key}`);
  }
});
test('all four landmark components reject all nonfinite numbers', () => {
  for (let index = 0; index < 4; index++) for (const value of [NaN, Infinity, -Infinity]) {
    const d = copy(); d.frames[0].lm[0][index] = value;
    assert.equal(validateCoords(d).ok, false);
  }
});
test('unknown fields rejected at all object levels', () => {
  for (const section of ['subject', 'capture', 'pose', 'marks']) {
    const d = copy(); d[section].unapproved = 1;
    assert.equal(validateCoords(d).ok, false, section);
  }
  const d = copy(); d.frames[0].unapproved = 1;
  assert.equal(validateCoords(d).ok, false);
});
test('JSON roundtrip preserves a valid synthetic payload', () => {
  const roundtrip = JSON.parse(JSON.stringify(fixture));
  assert.equal(validateCoords(roundtrip).ok, true);
  assert.deepEqual(roundtrip, fixture);
});
