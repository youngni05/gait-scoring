const {test}=require('node:test');
const assert=require('node:assert/strict');
const sample=require('../contracts/coords-1/synthetic.sample.json');
const {prepareGdiCandidate}=require('../.stage1-build/contracts/gdi-candidate-1/prepare.js');

test('maps anatomical sides in ankle/knee/hip/foot order and scales each image axis',()=>{
  const input=structuredClone(sample);
  const frame=input.frames.find(f=>f.lm);
  frame.lm=Array.from({length:33},(_,i)=>[i/40,i/50,-i/10,i/40]);
  const before=structuredClone(input),result=prepareGdiCandidate(input);
  const sides=result.frames[input.frames.indexOf(frame)].sides;
  for(const [side,indices] of [['left',[27,25,23,31]],['right',[28,26,24,32]]]) {
    ['ankle','knee','hip','footIndex'].forEach((name,i)=>assert.deepEqual(sides[side][name],{
      x:indices[i]/40*input.capture.width,y:indices[i]/50*input.capture.height,visibility:indices[i]/40,
    }));
  }
  assert.deepEqual(input,before);
  sides.left.ankle.x=999;
  assert.deepEqual(input,before);
});

test('preserves timestamps, missing frames and genuine boundary zeros without resampling',()=>{
  const input=structuredClone(sample),frame=input.frames.find(f=>f.lm);
  frame.lm[27]=[0,0,123,0];
  const result=prepareGdiCandidate(input);
  assert.deepEqual(result.frames.map(f=>f.t),input.frames.map(f=>f.t));
  assert.deepEqual(result.frames.map(f=>f.sides===null),input.frames.map(f=>f.lm===null));
  assert.deepEqual(result.frames[input.frames.indexOf(frame)].sides.left.ankle,{x:0,y:0,visibility:0});
  assert.equal(result.readyForInference,false);
  assert.equal(result.blockers.length,4);
  assert.equal('score' in result,false);
});

test('rejects invalid shape and nonincreasing timestamps through the source contract',()=>{
  assert.throws(()=>prepareGdiCandidate({}));
  const input=structuredClone(sample);input.frames[1].t=input.frames[0].t;
  assert.throws(()=>prepareGdiCandidate(input));
  const bad=structuredClone(sample);bad.frames.find(f=>f.lm).lm.pop();
  assert.throws(()=>prepareGdiCandidate(bad));
});

test('standing marks do not select a gait window or change candidate coordinates',()=>{
  const input=structuredClone(sample),expected=prepareGdiCandidate(input);
  input.marks={stand_start:null,stand_end:null};
  assert.deepEqual(prepareGdiCandidate(input),expected);
});
