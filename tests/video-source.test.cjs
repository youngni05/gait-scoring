const {test}=require('node:test');
const assert=require('node:assert/strict');
const {FrameAudit}=require('../.stage1-build/mobile/src/pose/frame-audit.js');
const {CompleteFrameAudit}=require('../.stage1-build/mobile/src/pose/complete-frame-audit.js');
test('frame audit preserves VFR media times and only counts gaps in presented numbers',()=>{
 const a=new FrameAudit();a.add(0.03,5);a.add(0.097,6);a.add(0.131,9);
 const s=a.snapshot();assert.deepEqual(s.recent,[0.03,0.097,0.131]);assert.equal(s.missedCallbacks,2);assert.equal(s.count,3);assert.equal(s.maxGap,0.097-0.03);
});
test('frame audit refuses invented invalid or nonmonotonic times',()=>{
 for(const value of [-1,NaN,Infinity]) assert.throws(()=>new FrameAudit().add(value,1));
 const a=new FrameAudit();a.add(1,1);assert.throws(()=>a.add(1,2));assert.throws(()=>a.add(0.9,2));assert.throws(()=>a.add(2,1));
});
test('frame audit memory is bounded and snapshots do not mutate source',()=>{
 const a=new FrameAudit();for(let i=0;i<10000;i++)a.add(i/30,i);const s=a.snapshot();assert.equal(s.recent.length,10);s.recent[0]=-1;assert.ok(a.snapshot().recent[0]>0);
});

test('offline audit sorts B-frame decode order and preserves exact VFR PTS',()=>{
 const a=new CompleteFrameAudit([0,0.1,1/30,0.2]);
 for(const t of [0,0.033333,0.1,0.2]) a.accept(t);
 a.finish();assert.equal(a.complete,true);assert.equal(a.expected[1],1/30);
});
test('offline audit refuses missing, extra, duplicated and changed decoded frames',()=>{
 for(const times of [[0,0.2],[0,0.1,0.2,0.3],[0,0.1,0.1],[0,0.10001,0.2]]) {
  const a=new CompleteFrameAudit([0,0.1,0.2]);
  assert.throws(()=>{for(const t of times)a.accept(t);a.finish()});assert.equal(a.complete,false);
 }
});
test('offline audit refuses ambiguous or invalid source timelines',()=>{
 for(const times of [[],[0,0],[0,-1],[NaN],[Infinity]])assert.throws(()=>new CompleteFrameAudit(times));
});
test('partial or cancelled iteration never implies completion',()=>{
 const a=new CompleteFrameAudit([0,1]);a.accept(0);assert.equal(a.complete,false);assert.equal(a.count,1);
});
