const {test}=require('node:test');
const assert=require('node:assert/strict');
const {prepareAnalysisMeasurement}=require('../.stage1-build/contracts/measurement-2/prepare.js');
const {mapResult}=require('../.stage1-build/legacy-ui/src/ui/data/apiSource.js');
const {fmtDate}=require('../.stage1-build/legacy-ui/src/ui/data/mock.js');
const fixture=require('../contracts/measurement-1/synthetic.request.json');
function request(){const b=structuredClone(fixture.body);const {schema_version,...metadata}=b.processing_metadata;return {coords:b.keypoints_json,metadata,requestId:crypto.randomUUID()};}
test('analysis request removes capture timestamp/source and freezes the start across retries',()=>{
 const r=request(), original=structuredClone(r), start='2026-01-01T15:00:00Z';
 const prepared=prepareAnalysisMeasurement(r,r.coords.subject.id,start,new Date('2026-01-02'));
 const body=JSON.parse(prepared.body);
 assert.equal(body.analysis_started_at,'2026-01-01T15:00:00.000Z');
 assert.equal(body.keypoints_json.schema_version,'coords-2');
 assert.equal(body.processing_metadata.schema_version,'processing-2');
 assert.ok(!('measured_at' in body));assert.ok(!('recorded_at' in body.keypoints_json.capture));
 assert.ok(!('recordedAtSource' in body.processing_metadata));
 assert.deepEqual(r,original);assert.ok(Object.isFrozen(prepared));
 assert.equal(fmtDate(body.analysis_started_at),'2026.01.02');
});
test('invalid or future analysis start cannot prepare a request',()=>{
 const r=request();for(const start of ['bad','2026-01-01','2026-01-03T00:00:00Z']) assert.throws(()=>prepareAnalysisMeasurement(r,r.coords.subject.id,start,new Date('2026-01-02')));
});
test('new result uses analysis start, legacy result retains its explicitly labelled capture date',()=>{
 const base={status:'not_available',participant:{height_cm:'160'},measurement:{id:'m',participant_id:'p',age:76,measured_at:null,analysis_started_at:'2026-01-01T15:00:00Z'},analysis_result:null};
 const r=mapResult(base);assert.equal(r.dateBasis,'analysis_start');assert.equal(r.measuredAt,base.measurement.analysis_started_at);assert.equal(r.ageAtCapture,76);
 const legacy=mapResult({...base,measurement:{...base.measurement,analysis_started_at:null,measured_at:'2020-01-01T00:00:00Z'}});
 assert.equal(legacy.dateBasis,'legacy_capture');assert.equal(legacy.measuredAt,'2020-01-01T00:00:00Z');
});
