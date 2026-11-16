const {test}=require('node:test');const assert=require('node:assert/strict');
const {FrontalStandDetector}=require('../.stage1-build/mobile/src/pose/stand-detector.js');
const signal=()=>new AbortController().signal;
function standing(t){const lm=Array.from({length:33},()=>[.5,.5,0,.99]);for(const [i,x,y] of [[11,.4,.2],[12,.6,.2],[23,.43,.45],[24,.57,.45],[25,.43,.65],[26,.57,.65],[27,.43,.85],[28,.57,.85],[29,.42,.87],[30,.58,.87],[31,.45,.89],[32,.55,.89]])lm[i]=[x+Math.sin(t*8)*.0005,y,0,.99];return {t,lm};}
function frames(n=91){return Array.from({length:n},(_,i)=>standing(i/30));}
const detect=(f)=>new FrontalStandDetector(1).analyze(f,10,signal()).detection;
test('stable upright interval uses observed endpoints and leaves raw frames untouched',()=>{const f=frames(),original=JSON.stringify(f);const d=detect(f);assert.equal(d.status,'detected');assert.deepEqual(d.marks,{stand_start:0,stand_end:3});assert.equal(JSON.stringify(f),original)});
test('walking from first frame does not invent initial two seconds',()=>{const f=frames();f.forEach((p,i)=>p.lm.forEach(l=>l[0]+=i*.004));assert.equal(detect(f).status,'not_detected')});
test('short pause is not enough',()=>assert.equal(detect(frames(31)).status,'not_detected'));
test('null and low visibility cannot form standing',()=>{for(const f of [frames().map(f=>({...f,lm:null})),frames().map(f=>{f.lm[27][3]=.1;return f})])assert.equal(detect(f).status,'not_detected')});
test('seated bent knees are rejected even if motionless',()=>{const f=frames();f.forEach(p=>{p.lm[25][0]=.65;p.lm[25][1]=.47;p.lm[26][0]=.8;p.lm[26][1]=.47});assert.equal(detect(f).status,'not_detected')});
test('frozen repeated coordinate output cannot pass',()=>{const f=frames().map(p=>({t:p.t,lm:standing(0).lm}));assert.equal(detect(f).status,'not_detected')});
test('gap and detection loss split intervals without interpolation',()=>{const f=frames(61);f[30].lm=null;assert.equal(detect(f).status,'not_detected');const g=frames(61);g.forEach((p,i)=>{if(i>=30)p.t+=.5});assert.equal(detect(g).status,'not_detected')});
test('rotation and camera scale changes are not stable standing',()=>{const f=frames();f.forEach((p,i)=>{p.lm[11][0]+=.002*i;p.lm[12][0]-=.002*i});assert.equal(detect(f).status,'not_detected')});
test('later stable interval is selected after walking, preserving VFR timestamps',()=>{const f=frames(151);f.forEach((p,i)=>{if(i<60)p.lm.forEach(l=>l[0]+=.004*i);p.t=i*.04});const d=detect(f);assert.equal(d.status,'detected');assert.ok(d.marks.stand_start>=2.4);assert.equal(d.marks.stand_end,6)});
test('invalid times and values throw rather than fabricating null success',()=>{const f=frames();f[2].t=0;assert.throws(()=>detect(f));const g=frames();g[0].lm[11][0]=NaN;assert.throws(()=>detect(g))});
test('aborted detection throws',()=>{const c=new AbortController();c.abort();assert.throws(()=>new FrontalStandDetector().analyze(frames(),3,c.signal),{name:'AbortError'})});
test('camera zoom cannot pass as still standing',()=>{const f=frames();f.forEach((p,i)=>p.lm.forEach(l=>{l[0]=.5+(l[0]-.5)*(1+i*.005);l[1]=.5+(l[1]-.5)*(1+i*.005)}));assert.equal(detect(f).status,'not_detected')});
test('out of view feet are insufficient evidence',()=>{const f=frames();f.forEach(p=>p.lm[31][1]=1.1);assert.equal(detect(f).status,'not_detected')});
