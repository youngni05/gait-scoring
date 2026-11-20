const {test} = require('node:test');
const assert = require('node:assert/strict');
const {ApiDataSource, mapSubject, mapResult} = require('../.stage1-build/mobile/src/ui/data/apiSource.js');
const {LoginFailedError} = require('../.stage1-build/mobile/src/ui/data/source.js');
const response = (data, status=200) => new Response(JSON.stringify(data), {status});
const person = {id:'11111111-1111-4111-8111-111111111111', name:'합성', gender:'F', birth_date:'1950-01-01', height_cm:'170.50'};
const result = {status:'available', participant:person, measurement:{id:'m1',participant_id:person.id,age:70,measured_at:'2020-01-01T00:00:00Z'},analysis_result:{gdi_score:0,cadence_spm:null,stride_length_cm:0,hip_graph_json:{left:[1],right:[2]},knee_graph_json:null,ankle_graph_json:null}};
test('result mapping preserves zero, null, measurement identity, age and current height',()=>{
 const r=mapResult(result); assert.equal(r.gdi,0); assert.equal(r.cadence,null); assert.equal(r.strideLength,0);
 assert.equal(r.id,'m1'); assert.equal(r.ageAtCapture,70); assert.equal(r.heightAtCapture,170.5);
 assert.equal(r.curves.hip,undefined); assert.match(r.missing.hip,/축·단위/); assert.ok(r.missing.cadence);
 const pending=mapResult({...result,status:'not_available',analysis_result:null});
 assert.equal(pending.status,'not_available'); assert.deepEqual(pending.missing,{});
 assert.equal(mapSubject({...person,gender:null,birth_date:null,height_cm:null}).heightCm,null);
});
test('graph mapping accepts only explicitly versioned finite 51-point cycle curves',()=>{
 const graph={schema_version:'gait-cycle-degrees-1',left:Array(51).fill(0),right:Array(51).fill(1)};
 assert.deepEqual(mapResult({...result,analysis_result:{...result.analysis_result,hip_graph_json:graph}}).curves.hip,{left:graph.left,right:graph.right});
 graph.left[0]=NaN;
 assert.equal(mapResult({...result,analysis_result:{...result.analysis_result,hip_graph_json:graph}}).curves.hip,undefined);
});
test('summary consumes every measurement page and sends bearer auth',async()=>{
 const offsets=[];
 const api=new ApiDataSource('http://localhost:8000',async(url,init)=>{
   if(url.endsWith('/auth/login')) return response({access_token:'t',token_type:'bearer'});
   assert.equal(init.headers.get('Authorization'),'Bearer t');
   if(url.endsWith('/participants')) return response([person]);
   const offset=Number(new URL(url).searchParams.get('offset')); offsets.push(offset);
   return response(Array.from({length:offset===0?20:1},(_,i)=>({id:String(offset+i),participant_id:person.id,measured_at:'2020-01-01T00:00:00Z',has_analysis_result:false})));
 });
 await api.login('worker','password'); const rows=await api.listSubjects();
 assert.equal(rows[0].resultCount,21);assert.deepEqual(offsets,[0,20]);
});
test('404 and wrong-subject result are absent, while network failures stay errors',async()=>{
 let mode='404';
 const api=new ApiDataSource('http://localhost:8000',async()=>{
   if(mode==='network') throw Error('offline');
   return mode==='404'?response({detail:'not found'},404):response(result);
 });
 assert.equal(await api.getSubject(person.id),null);
 mode='ok';assert.equal(await api.getResult('other','m1'),null);
 mode='network';await assert.rejects(api.getSubject(person.id),e=>e.code==='NETWORK');
});
test('renewal uses same account and failed login never becomes authenticated',async()=>{
 let ok=false;
 const api=new ApiDataSource('http://localhost:8000',async()=>ok?response({access_token:'t',token_type:'bearer'}):response({},401));
 await assert.rejects(api.login('worker','wrong'),LoginFailedError);
 ok=true;await api.login('worker','right');
 await assert.rejects(api.login('other','right'),/같은 계정/);
 api.logout();await api.login('other','right');
});
