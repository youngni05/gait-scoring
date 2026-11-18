const {test} = require('node:test');
const assert = require('node:assert/strict');
const {ApiClient, ApiError, normalizeBase, errorMessage} = require('../.stage1-build/mobile/src/api/client.js');
const fixture = require('../contracts/measurement-1/synthetic.request.json');
const participant = fixture.body.keypoints_json.subject.id;
const request = {method:'POST',path:`/participants/${participant}/measurements`,headers:{'Content-Type':'application/json','Idempotency-Key':'12345678-1234-1234-1234-123456789abc'},body:JSON.stringify(fixture.body)};
const receipt = {id:'measurement-1',participant_id:participant};
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
