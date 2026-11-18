const {test}=require('node:test');
const assert=require('node:assert/strict');
const {checkProcessingInput}=require('../.stage1-build/mobile/src/pose/processing-input.js');
const valid={subjectId:'DEV-TEST',heightCm:170,recordedAt:'2026-11-16T10:00:00+09:00',recordedAtSource:'user_entered'};
test('development input accepts explicit timestamp and source',()=>assert.deepEqual(checkProcessingInput(valid),[]));
for(const [field,value] of [['subjectId',' '],['heightCm',99],['heightCm',221],['heightCm',NaN],['recordedAt',''],['recordedAt','2026-02-30T10:00:00+09:00'],['recordedAt','2026-11-16T10:00:00'],['recordedAtSource','unknown']]){
 test(`development input rejects ${field}=${value}`,()=>assert.ok(checkProcessingInput({...valid,[field]:value}).length));
}

const {resolveRecordedAt,koreanMetadataTime}=require('../.stage1-build/mobile/src/pose/recorded-at.js');
test('Korean metadata time preserves instant across calendar boundary',()=>{
 assert.equal(koreanMetadataTime(new Date('2026-11-01T23:31:08Z')),'2026-11-02T08:31:08.000+09:00');
});
test('date override wins; clearing restores metadata, independent of host timezone',()=>{
 const meta='2026-11-01T14:31:08.000+09:00';
 assert.deepEqual(resolveRecordedAt('2026-11-03',meta),{recordedAt:'2026-11-03T00:00:00+09:00',recordedAtSource:'user_entered'});
 assert.deepEqual(resolveRecordedAt('',meta),{recordedAt:meta,recordedAtSource:'metadata'});
 assert.equal(resolveRecordedAt('',null).recordedAt,'');
});
test('invalid manual dates never silently fall back; leap days validated',()=>{
 for(const day of ['2026-02-30','2026-02-29','2026-13-01','2026/11/01']) assert.equal(resolveRecordedAt(day,'2026-11-01T00:00:00Z').recordedAt,'');
 assert.equal(resolveRecordedAt('2024-02-29',null).recordedAt,'2024-02-29T00:00:00+09:00');
});
test('missing invalid and sentinel metadata require user date',()=>{
 for(const d of [null,undefined,new Date(NaN),new Date('1904-01-01T00:00:00Z')]) assert.equal(koreanMetadataTime(d),null);
});
