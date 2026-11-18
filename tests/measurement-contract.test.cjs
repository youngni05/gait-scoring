const {test}=require('node:test');
const assert=require('node:assert/strict');
const sample=require('../contracts/coords-1/synthetic.sample.json');
const {prepareMeasurement}=require('../.stage1-build/contracts/measurement-1/prepare.js');
const participant='22222222-2222-4222-8222-222222222222';
const now=new Date('2030-01-01T00:00:00Z');
function request(){
  const coords=structuredClone(sample); coords.subject.id=participant;
  return {coords,requestId:'44444444-4444-4444-8444-444444444444',metadata:{
    recordedAtSource:'user_entered',rotationAppliedDegrees:0,
    detection:{status:'detected',marks:coords.marks,version:'fixture-1'},
    modelAssetVersion:'pose_landmarker_full/float16/1',
  }};
}
test('mapping preserves coordinates, PTS, null frames, date provenance and model identity',()=>{
  const r=request(),before=structuredClone(r),p=prepareMeasurement(r,participant,now),b=JSON.parse(p.body);
  assert.equal(p.path,`/participants/${participant}/measurements`);
  assert.equal(p.headers['Idempotency-Key'],r.requestId);
  assert.deepEqual(b.keypoints_json,r.coords);
  assert.deepEqual(b.processing_metadata,{schema_version:'processing-1',...r.metadata});
  assert.equal(b.extractor_name,'mediapipe_tasks_vision');
  assert.equal(b.extractor_version,r.coords.pose.engine_version);
  assert.equal(b.measured_at,r.coords.capture.recorded_at);
  assert.deepEqual(Object.keys(b).sort(),['extractor_name','extractor_version','keypoints_json','measured_at','processing_metadata']);
  assert.deepEqual(r,before);
});
test('prepared bytes and headers remain stable after UI changes and across retries',()=>{
  const r=request(),p=prepareMeasurement(r,participant,now),bytes=p.body;
  r.coords.subject.height_cm=180; r.metadata.recordedAtSource='metadata'; r.requestId='new';
  assert.equal(p.body,bytes);assert.equal(p.headers['Idempotency-Key'],'44444444-4444-4444-8444-444444444444');
  assert.ok(Object.isFrozen(p));assert.ok(Object.isFrozen(p.headers));
});
test('rejects selected-subject mismatch and development IDs',()=>{
  assert.throws(()=>prepareMeasurement(request(),'33333333-3333-4333-8333-333333333333',now));
  const r=request();r.coords.subject.id='DEV-EXAMPLE';assert.throws(()=>prepareMeasurement(r,'DEV-EXAMPLE',now));
});
test('rejects invalid retry keys and future captures',()=>{
  const r=request();r.requestId='header\r\ninjection';assert.throws(()=>prepareMeasurement(r,participant,now));
  const s=request();assert.throws(()=>prepareMeasurement(s,participant,new Date('2000-01-01Z')));
});
test('rejects metadata loss, invalid versions, mismatched detection and private paths',()=>{
  for(const alter of [r=>r.metadata.detection.version=' ',r=>r.coords.pose.engine_version='x'.repeat(101),
    r=>r.metadata.modelAssetVersion='',r=>r.metadata.localPath='C:/private.mp4',
    r=>r.metadata.detection.marks={stand_start:null,stand_end:null}]){
    const r=request();alter(r);assert.throws(()=>prepareMeasurement(r,participant,now));
  }
});
test('same request produces same serialized payload without network calls',()=>{
  const r=request();assert.deepEqual(prepareMeasurement(r,participant,now),prepareMeasurement(r,participant,now));
});
test('published synthetic request matches the adapter output',()=>{
  const fixture=require('../contracts/measurement-1/synthetic.request.json');
  const p=prepareMeasurement(request(),participant,now);
  assert.deepEqual({...p,body:JSON.parse(p.body)},fixture);
});
test('missing standing detection preserves explicit null marks and reason',()=>{
  const r=request();r.coords.marks={stand_start:null,stand_end:null};
  r.metadata.detection={status:'not_detected',marks:r.coords.marks,version:'fixture-1',reason:'insufficient evidence'};
  const b=JSON.parse(prepareMeasurement(r,participant,now).body);
  assert.deepEqual(b.keypoints_json.marks,r.coords.marks);
  assert.equal(b.processing_metadata.detection.reason,'insufficient evidence');
});
