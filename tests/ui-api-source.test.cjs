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
test('subject list uses one summary request, regardless of history size',async()=>{
 const calls=[];
 const api=new ApiDataSource('http://localhost:8000',async(url,init)=>{
   calls.push(new URL(url).pathname);
   if(url.endsWith('/auth/login')) return response({access_token:'t',token_type:'bearer'});
   assert.equal(init.headers.get('Authorization'),'Bearer t');
   assert.ok(url.endsWith('/participants/summaries'));
   return response([{...person,measurement_count:10000,latest_measured_at:'2020-01-01T00:00:00Z'}]);
 });
 await api.login('worker','password'); const rows=await api.listSubjects();
 assert.equal(rows[0].resultCount,10000);
 assert.deepEqual(calls,['/auth/login','/participants/summaries']);
});
test('result list reads only the requested summary page and preserves zero/null',async()=>{
 const urls=[];
 const api=new ApiDataSource('http://localhost:8000',async(url)=>{
   urls.push(url);
   return response({items:[{...result.measurement,has_analysis_result:true,gdi_score:0,cadence_spm:null,stride_length_cm:0}],total:21,offset:20,limit:20});
 });
 const page=await api.listResults(person.id,undefined,20);
 assert.equal(page.total,21); assert.equal(page.items.length,1);
 assert.equal(page.items[0].gdi,0); assert.equal(page.items[0].cadence,null);
 assert.deepEqual(page.items[0].curves,{});
 assert.equal(urls.length,1);
 assert.ok(urls[0].endsWith('/measurement-summaries?limit=20&offset=20'));
});
test('detail reads a summary without reading measurements or graphs',async()=>{
 let calls=0;
 const api=new ApiDataSource('http://localhost:8000',async(url)=>{
   calls++; assert.ok(url.endsWith('/'+person.id+'/summary'));
   return response({...person,measurement_count:42,latest_measured_at:null});
 });
 assert.equal((await api.getSubjectSummary(person.id)).resultCount,42); assert.equal(calls,1);
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

const {LeaveGuards}=require('../.stage1-build/mobile/src/ui/leave-guards.js');
test('shared leave guard keeps pending work on rejection and unregisters on exit',()=>{
 const guards=new LeaveGuards(); let pending=true, prompts=0;
 const remove=guards.register(()=>pending?'pending save':null);
 const reject=message=>{prompts++;assert.equal(message,'pending save');return false;};
 assert.equal(guards.confirm(reject),false);
 assert.equal(guards.confirm(()=>true),true);
 assert.equal(guards.message(),'pending save'); // approval alone does not drop the request
 pending=false;assert.equal(guards.confirm(reject),true);assert.equal(prompts,1);
 pending=true;remove();assert.equal(guards.message(),null);
});
