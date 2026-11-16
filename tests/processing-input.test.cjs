const {test}=require('node:test');
const assert=require('node:assert/strict');
const {checkProcessingInput}=require('../.stage1-build/mobile/src/pose/processing-input.js');
const valid={subjectId:'DEV-TEST',heightCm:170,recordedAt:'2026-11-16T10:00:00+09:00',recordedAtSource:'user_entered'};
test('development input accepts explicit timestamp and source',()=>assert.deepEqual(checkProcessingInput(valid),[]));
for(const [field,value] of [['subjectId',' '],['heightCm',99],['heightCm',221],['heightCm',NaN],['recordedAt',''],['recordedAt','2026-02-30T10:00:00+09:00'],['recordedAt','2026-11-16T10:00:00'],['recordedAtSource','unknown']]){
 test(`development input rejects ${field}=${value}`,()=>assert.ok(checkProcessingInput({...valid,[field]:value}).length));
}
