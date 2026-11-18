const {test}=require('node:test');const assert=require('node:assert/strict');
const {toPoseFrame}=require('../.stage1-build/mobile/src/pose/pose-values.js');
const {extractVideo}=require('../.stage1-build/mobile/src/pose/extract-video.js');
const pose=()=>Array.from({length:33},(_,i)=>({x:-0.01+i/30,y:1.01,z:-.123456789,visibility:.8}));
test('preserves all 33 original values and timestamp without clamping or rounding',()=>{const raw=pose();const f=toPoseFrame(.0333667,[raw]);assert.equal(f.t,.0333667);assert.deepEqual(f.lm[0],[-.01,1.01,-.123456789,.8]);raw[0].x=5;assert.equal(f.lm[0][0],-.01);});
test('no detection is null',()=>assert.deepEqual(toPoseFrame(0,[]),{t:0,lm:null}));
test('malformed, multiple and nonfinite poses are errors, never fabricated coordinates',()=>{
 assert.throws(()=>toPoseFrame(0,[pose(),pose()]));assert.throws(()=>toPoseFrame(0,[pose().slice(1)]));
 for(const key of ['x','y','z','visibility']) {const p=pose();p[0][key]=NaN;assert.throws(()=>toPoseFrame(0,[p]));}
 const p=pose();delete p[0].visibility;assert.throws(()=>toPoseFrame(0,[p]));
});
function source(times){let released=0,closed=false;return {get released(){return released},get closed(){return closed},async *frames(){try{for(const t of times)yield{t,pixels:{},release(){released++}}}finally{closed=true}}};}
test('pipeline preserves VFR and null frames, releases every input',async()=>{const s=source([0,.033,.1]);const r=await extractVideo(s,{async extract(f){return toPoseFrame(f.t,f.t===.033?[]:[pose()])}},new AbortController().signal);assert.deepEqual(r.frames.map(f=>f.t),[0,.033,.1]);assert.equal(r.detected,2);assert.equal(r.missing,1);assert.equal(s.released,3);assert.ok(s.closed)});
test('cancel releases current frame and does not return a complete partial result',async()=>{const c=new AbortController(),s=source([0,1,2]);await assert.rejects(extractVideo(s,{async extract(f){c.abort();return {t:f.t,lm:null}}},c.signal),{name:'AbortError'});assert.equal(s.released,1);assert.ok(s.closed)});
test('engine error releases source and propagates',async()=>{const s=source([0]);await assert.rejects(extractVideo(s,{async extract(){throw Error('engine failure')}},new AbortController().signal),/engine failure/);assert.equal(s.released,1);assert.ok(s.closed)});
test('pipeline rejects altered timestamp',async()=>{const s=source([0]);await assert.rejects(extractVideo(s,{async extract(){return {t:1,lm:null}}},new AbortController().signal),/시각/);assert.equal(s.released,1)});
