const {test} = require('node:test');
const assert = require('node:assert/strict');
const {ApiClient, ApiError, normalizeBase, errorMessage} = require('../.stage1-build/mobile/src/api/client.js');
const fixture = require('../contracts/measurement-1/synthetic.request.json');
const participant = fixture.body.keypoints_json.subject.id;
const request = {method:'POST',path:`/participants/${participant}/measurements`,headers:{'Content-Type':'application/json','Idempotency-Key':'12345678-1234-1234-1234-123456789abc'},body:JSON.stringify(fixture.body)};
const receipt = {id:'measurement-1',participant_id:participant};
const {validateParticipant, participantChanges, koreaToday} = require('../.stage1-build/mobile/src/api/participant-input.js');
const personInput = {name:'  테스트 대상자  ',gender:'F',birth_date:'1950-01-01',height_cm:'170.5'};
const response = (data,status=200,headers={}) => new Response(JSON.stringify(data),{status,headers});

test('API URL rejects credentials, query, non-HTTP and insecure external endpoints',()=>{
  assert.equal(normalizeBase(' https://example.com/api/ '),'https://example.com/api');
  assert.equal(normalizeBase('http://localhost:8000'),'http://localhost:8000');
  for(const url of ['file:///tmp/test','https://user:secret@example.com','https://example.com?token=secret','https://example.com#x','http://example.com']) assert.throws(()=>normalizeBase(url));
});
test('login sends form encoded credentials without existing bearer or cookies',async()=>{
  let call;
  const api=new ApiClient('https://example.com',()=> 'old',async(url,init)=>{call={url,init};return response({access_token:'fresh',token_type:'bearer'});});
  assert.equal(await api.login('a+b','p&=한글'),'fresh');
  assert.equal(call.url,'https://example.com/auth/login');
  assert.equal(new URLSearchParams(call.init.body).get('password'),'p&=한글');
  assert.equal(call.init.headers.has('Authorization'),false);
  assert.equal(call.init.credentials,'omit'); assert.equal(call.init.redirect,'error');
});
test('network loss after possible save retries exact body and key',async()=>{
  const calls=[];
  const api=new ApiClient('https://example.com',()=> 'token',async(url,init)=>{calls.push(init);if(calls.length===1)throw Error('lost response');return response(receipt,200);});
  await assert.rejects(api.save(request),e=>e.code==='NETWORK');
  assert.deepEqual(await api.save(request),receipt);
  assert.equal(calls[0].body,calls[1].body);
  assert.equal(calls[0].headers.get('Idempotency-Key'),calls[1].headers.get('Idempotency-Key'));
  assert.equal(calls[0].body,request.body);
});
test('401 followed by token renewal preserves request bytes and request key',async()=>{
  let token='expired';const calls=[];
  const api=new ApiClient('https://example.com',()=>token,async(url,init)=>{calls.push(init);return token==='expired'?response({detail:'Invalid token'},401):response(receipt,201);});
  await assert.rejects(api.save(request),e=>e.status===401);
  token='renewed';assert.deepEqual(await api.save(request),receipt);
  assert.equal(calls[1].headers.get('Authorization'),'Bearer renewed');
  assert.equal(calls[0].body,calls[1].body);
});
test('conflict and retryable lock errors preserve machine code and delay',async()=>{
  for(const code of ['IDEMPOTENCY_CONFLICT','REQUEST_IN_PROGRESS']){
    const api=new ApiClient('https://example.com',()=>'',async()=>response({detail:{code,message:'private raw input'}},409,{'Retry-After':'2'}));
    await assert.rejects(api.save(request),e=>e instanceof ApiError && e.code===code && e.retryAfter===2 && !errorMessage(e).includes('private'));
  }
});
test('422 string, array and structured errors are handled without echoing server input',async()=>{
  for(const detail of ['private', [{msg:'private',input:'private'}],{code:'PARTICIPANT_MISMATCH',message:'private'}]){
    const api=new ApiClient('https://example.com',()=>'',async()=>response({detail},422));
    await assert.rejects(api.save(request),e=>e.status===422 && !errorMessage(e).includes('private'));
  }
});
test('unexpected success status, wrong participant and malformed JSON do not yield receipt',async()=>{
  for(const get of [()=>response(receipt,202),()=>response({...receipt,participant_id:'wrong'},201),()=>response({},201),()=>new Response('<html>gateway</html>',{status:200})]){
    const api=new ApiClient('https://example.com',()=>'',async()=>get());
    await assert.rejects(api.save(request),e=>e.code==='INVALID_RESPONSE');
  }
});
test('timeout and caller cancellation terminate transport and allow later retry',async()=>{
  const transport=async(url,init)=>new Promise((resolve,reject)=>{if(init.signal.aborted)reject(Error('abort'));else init.signal.addEventListener('abort',()=>reject(Error('abort')),{once:true});});
  const api=new ApiClient('https://example.com',()=>'',transport,10);
  await assert.rejects(api.save(request),e=>e.code==='NETWORK');
  const abort=new AbortController();abort.abort();
  await assert.rejects(api.save(request,abort.signal),e=>e.code==='CANCELLED');
});
test('participant list accepts Decimal strings and missing profile fields',async()=>{
  const rows=[{id:participant,name:'Synthetic',height_cm:'170.00',birth_date:null}];
  const api=new ApiClient('https://example.com',()=>'',async()=>response(rows));
  assert.deepEqual(await api.participants(),rows);
});
test('history uses pagination; missing analysis is not invented as a score or job state',async()=>{
  const urls=[];
  const api=new ApiClient('https://example.com',()=>'',async(url)=>{urls.push(url);return url.includes('/result')?response({status:'not_available',measurement:{id:'m1'},analysis_result:null}):response([]);});
  assert.deepEqual(await api.measurements(participant,20),[]);
  assert.ok(urls[0].endsWith('?limit=20&offset=20'));
  const result=await api.result('m1');assert.equal(result.status,'not_available');assert.equal(result.analysis_result,null);
});

test('participant input trims names and retains decimal precision without rounding invalid input',()=>{
  const result=validateParticipant(personInput,'2026-11-18');
  assert.deepEqual(result.body,{name:'테스트 대상자',gender:'F',birth_date:'1950-01-01',height_cm:'170.50'});
  for(const value of ['', '0','-1','1e2','NaN','Infinity','170.001','1000','170,5']) assert.ok(validateParticipant({...personInput,height_cm:value}).errors.height_cm);
  assert.ok(validateParticipant({...personInput,height_cm:'99.99'}).body); // profile schema, not pose schema
  assert.ok(validateParticipant({...personInput,height_cm:'999.99'}).body);
});
test('participant dates use Korea today and reject rollover, zero year and future dates',()=>{
  assert.equal(koreaToday(new Date('2026-11-17T15:00:00Z')),'2026-11-18');
  for(const birth_date of ['', '1950-02-30','1900-02-29','0000-01-01','2026-11-19','1950-1-1']) assert.ok(validateParticipant({...personInput,birth_date},'2026-11-18').errors.birth_date);
  for(const birth_date of ['2000-02-29','2026-11-18']) assert.ok(validateParticipant({...personInput,birth_date},'2026-11-18').body);
});
test('participant name and gender are required; Unicode length agrees with backend',()=>{
  for(const name of ['  ','가'.repeat(101)]) assert.ok(validateParticipant({...personInput,name}).errors.name);
  assert.ok(validateParticipant({...personInput,name:'😀'.repeat(100)}).body);
  assert.ok(validateParticipant({...personInput,gender:''}).errors.gender);
});
test('profile patch excludes unchanged numeric formatting, IDs and measurement fields',()=>{
  const body=validateParticipant(personInput).body;
  const original={...body,id:participant,height_cm:'170.500',age:76,keypoints_json:{kept:true}};
  assert.deepEqual(participantChanges(original,body),{});
  assert.deepEqual(participantChanges(original,{...body,birth_date:'1951-01-01'}),{birth_date:'1951-01-01'});
  assert.equal(original.age,76); assert.deepEqual(original.keypoints_json,{kept:true});
  assert.deepEqual(participantChanges({...original,height_cm:null},body),{height_cm:'170.50'});
});
test('participant creation sends authenticated JSON to POST and accepts validated receipt',async()=>{
  const body=validateParticipant(personInput).body,person={...body,id:participant};let call;
  const api=new ApiClient('https://example.com',()=> 'token',async(url,init)=>{call={url,init};return response(person,201);});
  assert.deepEqual(await api.createParticipant(body),person);
  assert.equal(call.url,'https://example.com/participants');assert.equal(call.init.method,'POST');
  assert.equal(call.init.headers.get('Authorization'),'Bearer token');
  assert.equal(call.init.headers.get('Content-Type'),'application/json');
  assert.equal(call.init.headers.has('Idempotency-Key'),false);
  assert.deepEqual(JSON.parse(call.init.body),body);
});
test('participant updates use PATCH with changed fields only and reject no-op before transport',async()=>{
  const body=validateParticipant(personInput).body;let calls=[];
  const api=new ApiClient('https://example.com',()=>'',async(url,init)=>{calls.push({url,init});return response({...body,id:participant});});
  await assert.rejects(api.updateParticipant(participant,{}));assert.equal(calls.length,0);
  await api.updateParticipant(participant,{name:'수정 이름'});
  assert.equal(calls[0].init.method,'PATCH');assert.ok(calls[0].url.endsWith('/participants/'+participant));
  assert.deepEqual(JSON.parse(calls[0].init.body),{name:'수정 이름'});
});
test('participant write rejects wrong ID, missing profile and unexpected status',async()=>{
  const body=validateParticipant(personInput).body;
  for(const data of [{}, {...body,id:participant,height_cm:null}, {...body,id:'invalid'}, {...body,id:participant,gender:'X'}]){
    const api=new ApiClient('https://example.com',()=>'',async()=>response(data,201));
    await assert.rejects(api.createParticipant(body),e=>e.code==='INVALID_RESPONSE');
  }
  const api=new ApiClient('https://example.com',()=>'',async()=>response({...body,id:'11111111-1111-1111-1111-111111111111'}));
  await assert.rejects(api.updateParticipant(participant,{name:'수정'}),e=>e.code==='INVALID_RESPONSE');
  const accepted=new ApiClient('https://example.com',()=>'',async()=>response({...body,id:participant},202));
  await assert.rejects(accepted.createParticipant(body),e=>e.code==='INVALID_RESPONSE');
});
test('participant creation never automatically retries after a lost response',async()=>{
  let calls=0;const api=new ApiClient('https://example.com',()=>'',async()=>{calls++;throw Error('lost');});
  await assert.rejects(api.createParticipant(validateParticipant(personInput).body),e=>e.code==='NETWORK');
  assert.equal(calls,1);
});
