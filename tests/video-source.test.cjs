const {test}=require('node:test');
const assert=require('node:assert/strict');
const {FrameAudit}=require('../.stage1-build/mobile/src/pose/frame-audit.js');
const {BrowserVideoSession,BrowserVideoSource}=require('../.stage1-build/mobile/src/pose/browser-video-source.js');
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
class Video extends EventTarget {
 currentTime=0;ended=false;controls=true;videoWidth=320;videoHeight=180;duration=2;callbacks=new Map();next=0;removed=false;
 pause(){} play(){return Promise.resolve()} load(){} removeAttribute(){} remove(){this.removed=true}
 requestVideoFrameCallback(cb){this.callbacks.set(++this.next,cb);return this.next}
 cancelVideoFrameCallback(id){this.callbacks.delete(id)}
 emitFrame(t,p){const [id,cb]=this.callbacks.entries().next().value;this.callbacks.delete(id);cb(999999,{mediaTime:t,presentedFrames:p})}
}
const info={width:320,height:180,durationSeconds:2,orientation:'landscape',rotationAppliedDegrees:null};
function setup(){global.document={createElement:()=>({width:0,height:0,getContext:()=>({drawImage(){}})})};return new Video();}
test('sequential source uses mediaTime, reuses one buffer, and closes callbacks on end',async()=>{
 const v=setup(),s=new BrowserVideoSession(v,info,'blob:test');const iter=s.frames(new AbortController().signal)[Symbol.asyncIterator]();
 let p=iter.next();v.emitFrame(0.01,1);const a=await p;assert.equal(a.value.t,0.01);a.value.release();
 p=iter.next();v.emitFrame(0.077,2);const b=await p;assert.equal(b.value.t,0.077);assert.equal(a.value.pixels,b.value.pixels);b.value.release();
 p=iter.next();v.ended=true;v.dispatchEvent(new Event('ended'));assert.equal((await p).done,true);assert.equal(v.callbacks.size,0);assert.equal(a.value.pixels.width,0);
});
test('abort during pending frame clears callback and allows retry',async()=>{
 const v=setup(),s=new BrowserVideoSession(v,info,'blob:test'),c=new AbortController();const i=s.frames(c.signal)[Symbol.asyncIterator]();const p=i.next();c.abort();await assert.rejects(p,{name:'AbortError'});assert.equal(v.callbacks.size,0);
 const j=s.frames(new AbortController().signal)[Symbol.asyncIterator]();const q=j.next();v.emitFrame(0,1);assert.equal((await q).value.t,0);await j.return();assert.equal(s.audit.snapshot().count,1);
});
test('dispose cancels pending frame and releases video once',async()=>{
 const v=setup(),s=new BrowserVideoSession(v,info,'blob:test');const iter=s.frames(new AbortController().signal)[Symbol.asyncIterator]();const p=iter.next();const rejection=assert.rejects(p,{name:'AbortError'});await s.dispose();await rejection;await s.dispose();assert.equal(v.removed,true);assert.equal(v.callbacks.size,0);
});
test('unsupported timestamp API rejects without fabricating FPS',async()=>{
 const v=setup();v.requestVideoFrameCallback=undefined;const s=new BrowserVideoSession(v,info,'blob:test');await assert.rejects(s.frames(new AbortController().signal)[Symbol.asyncIterator]().next(),/지원하지/);
});
test('empty files are rejected before allocating object URL',async()=>{
 await assert.rejects(new BrowserVideoSource({}).open(new File([],'empty.mp4'),new AbortController().signal),/비어 있지/);
});
for(const mode of ['decode-error','abort']) test(`opening ${mode} revokes object URL and removes decoder`,async()=>{
 const create=URL.createObjectURL,revoke=URL.revokeObjectURL;let revoked=0;
 const v=new Video();v.setAttribute=()=>{};
 Object.defineProperty(v,'src',{set(){if(mode==='decode-error')queueMicrotask(()=>v.dispatchEvent(new Event('error')))}});
 global.document={createElement:()=>v};URL.createObjectURL=()=> 'blob:fixture';URL.revokeObjectURL=url=>{assert.equal(url,'blob:fixture');revoked++};
 try{
  const c=new AbortController();const promise=new BrowserVideoSource({replaceChildren(){}}).open(new File(['synthetic'],'test.mp4'),c.signal);
  const rejected=assert.rejects(promise);if(mode==='abort')c.abort();await rejected;
  assert.equal(revoked,1);assert.equal(v.removed,true);
 }finally{URL.createObjectURL=create;URL.revokeObjectURL=revoke;}
});
test('frame draw failure terminates and releases canvas',async()=>{
 const v=setup();global.document={createElement:()=>({width:0,height:0,getContext:()=>({drawImage(){throw new Error('synthetic draw failure')}})})};
 const s=new BrowserVideoSession(v,info,'blob:test'),i=s.frames(new AbortController().signal)[Symbol.asyncIterator]();
 const p=i.next();v.emitFrame(0,1);await assert.rejects(p,/draw failure/);assert.equal(v.callbacks.size,0);assert.equal(v.controls,true);
});
test('browser drop statistics use per-run counters rather than claimed source frame count',async()=>{
 const v=setup();let quality={droppedVideoFrames:3,totalVideoFrames:20};v.getVideoPlaybackQuality=()=>quality;
 const s=new BrowserVideoSession(v,info,'blob:test'),i=s.frames(new AbortController().signal)[Symbol.asyncIterator]();
 const p=i.next();v.emitFrame(0,1);await p;quality={droppedVideoFrames:5,totalVideoFrames:30};
 assert.deepEqual(s.playbackQuality(),{dropped:2,total:10});await i.return();
 v.currentTime=1;const controller=new AbortController();const restart=s.frames(controller.signal)[Symbol.asyncIterator]();const pending=restart.next();controller.abort();await assert.rejects(pending,{name:'AbortError'});assert.equal(s.playbackQuality(),null);assert.equal(s.audit.snapshot().count,0);
});
