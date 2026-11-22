import {BrowserVideoSource}from'../../mobile/src/pose/browser-video-source';
import {MediaPipePoseExtractor}from'../../mobile/src/pose/mediapipe-extractor';
import {extractVideo}from'../../mobile/src/pose/extract-video';
import {FrontalStandDetector}from'../../mobile/src/pose/stand-detector';
import {JsonCoordsSerializer}from'../../mobile/src/pose/coords-serializer';
import {MockUploadGateway,type MockScenario,type UploadRequest}from'../../mobile/src/pose/upload-gateway';
const assert=(ok:unknown,m:string)=>{if(!ok)throw Error(m)};let outputUrl:string|null=null;
window.addEventListener('pagehide',()=>{if(outputUrl)URL.revokeObjectURL(outputUrl)});
document.querySelector<HTMLButtonElement>('#run')!.onclick=async()=>{
 const button=document.querySelector<HTMLButtonElement>('#run')!,download=document.querySelector<HTMLButtonElement>('#download')!,out=document.querySelector('#result')!;
 button.disabled=true;download.disabled=true;out.textContent='';if(outputUrl){URL.revokeObjectURL(outputUrl);outputUrl=null;}
 const log=(s:string)=>{out.textContent+=s+'\n'};const signal=new AbortController().signal; const started=performance.now();
 let session:Awaited<ReturnType<BrowserVideoSource['open']>>|undefined,engine:MediaPipePoseExtractor|undefined;
 try{
  const files=[...document.querySelector<HTMLInputElement>('#files')!.files!],file=files.find(f=>f.name.endsWith('.mp4'))!,expected:number[]=JSON.parse(await files.find(f=>f.name.endsWith('.json'))!.text());
  log(`Browser: ${navigator.userAgent}`);
  session=await new BrowserVideoSource(document.querySelector<HTMLElement>('#preview')!).open(file,signal);log(`Video: ${JSON.stringify(session.info)}; expected frames=${expected.length}`);
  const initStart=performance.now();engine=await MediaPipePoseExtractor.create(signal);log(`Model init ms=${(performance.now()-initStart).toFixed(1)}`);
  const run=await extractVideo(session,engine,signal,n=>{if(n%100===0)log(`processing ${n}/${expected.length}`)});await engine.dispose();engine=undefined;
  log(`Full extraction ms=${run.elapsedMs.toFixed(1)}; detected=${run.detected}; missing=${run.missing}; missing percent=${(100*run.missing/run.frames.length).toFixed(3)}`);
  assert(session.verification?.complete&&run.frames.length===expected.length,'Incomplete source');
  const original=JSON.stringify(run.frames),detection=new FrontalStandDetector(session.info.width/session.info.height).analyze(run.frames,session.info.durationSeconds,signal).detection;
  const serializer=new JsonCoordsSerializer(),recordingId=crypto.randomUUID(),requestId=crypto.randomUUID();
  const created=serializer.create({input:{subjectId:'DEV-QA-ONLY',heightCm:170,recordedAt:'2000-01-01T00:00:00+09:00',recordedAtSource:'user_entered',recordingId},video:session.info,frames:run.frames,marks:detection.marks,pose:{model:'pose_landmarker_full',engine_version:'1.2.3',delegate:'CPU'}});
  if(!created.ok)throw Error(created.issues.map(i=>i.message).join('; '));
  const text=serializer.stringify(created.value),coords=JSON.parse(text)as typeof created.value;
  assert(JSON.stringify(run.frames)===original,'Raw output was modified');
  coords.frames.forEach((f,i)=>{const raw=run.frames[i]!;assert(f.t===raw.t&&Math.abs(f.t-expected[i]!)<1e-8,'PTS changed');if(raw.lm===null)assert(f.lm===null,'null replaced');else{assert(f.lm?.length===33,'Not33');raw.lm.forEach((p,j)=>p.forEach((v,k)=>assert(f.lm![j]![k]===Number(v.toFixed(5)),'Coordinate round mismatch')))}});
  assert(coords.capture.recorded_at==='2000-01-01T00:00:00+09:00'&&coords.subject.id==='DEV-QA-ONLY'&&coords.subject.height_cm===170,'Input snapshot mismatch');
  const request:UploadRequest={coords,requestId,metadata:{recordedAtSource:'user_entered',rotationAppliedDegrees:session.info.rotationAppliedDegrees,detection,modelAssetVersion:'pose_landmarker_full/float16/1'}};
  const body=JSON.stringify(request),bytes=new TextEncoder().encode(text).length;
  assert(!body.includes(file.name)&&!body.includes('data:image')&&!body.includes('blob:')&&!body.includes('base64'),'Media locator in request');
  log(`PASS coords-1 ${coords.frames.length}/${expected.length}; null=${coords.frames.filter(f=>!f.lm).length}; ALL original PTS matched; ALL coordinates rounded <=5 decimals; raw unchanged`);
  log(`PASS subject/height/test-date snapshot; marks=${JSON.stringify(coords.marks)}; metadata separate; body allowlist validated by gateway`);
  log(`JSON bytes=${bytes}; source bytes=${file.size}; size comparison is NOT the no-upload proof`);
  let scenario:MockScenario='timeout';const gateway=new MockUploadGateway(()=>scenario,15);const immutable=JSON.stringify(request);
  for(const mode of ['timeout','network','auth']as const){scenario=mode;let failed=false;try{await gateway.submit(request,signal)}catch(e){failed=e instanceof Error&&'code'in e&&e.code===mode}assert(failed,`Missing ${mode}`);log(`PASS mock ${mode}`)}
  const abort=new AbortController();const pending=gateway.submit(request,abort.signal);abort.abort();let cancelled=false;try{await pending}catch(e){cancelled=e instanceof DOMException&&e.name==='AbortError'}assert(cancelled,'Cancel not handled');
  scenario='success';const first=await gateway.submit(request,signal),retry=await gateway.submit(request,signal);assert(first.environment==='mock'&&first.receiptId===retry.receiptId&&JSON.stringify(request)===immutable,'Retry changed payload/IDs/receipt');
  log('PASS cancellation; successful retry with SAME recording_id, requestId, body and mock receipt');
  log(`Primary pipeline plus JSON/mock ms=${(performance.now()-started).toFixed(1)}`);
  const cancel=new AbortController();engine=await MediaPipePoseExtractor.create(signal);let cancelCount=0,cancelledRun=false;
  try{await extractVideo(session,engine,cancel.signal,n=>{cancelCount=n;if(n===17)cancel.abort()})}catch(e){cancelledRun=e instanceof DOMException&&e.name==='AbortError'}
  assert(cancelledRun&&cancelCount===17,'Extraction cancellation failed');await engine.dispose();engine=undefined;
  log('PASS extraction cancelled at exactly 17; incomplete result rejected');
  engine=await MediaPipePoseExtractor.create(signal);
  const restarted=await extractVideo(session,engine,signal,n=>{if(n%100===0)log(`restart ${n}/${expected.length}`)});
  await engine.dispose();engine=undefined;
  assert(session.verification?.complete&&restarted.frames.length===expected.length,'Restart incomplete');
  restarted.frames.forEach((f,i)=>assert(f.t===run.frames[i]!.t&&Math.abs(f.t-expected[i]!)<1e-8,'Restart PTS mismatch'));
  log(`PASS restart ALL PTS; ms=${restarted.elapsedMs.toFixed(1)}; detected=${restarted.detected}; missing=${restarted.missing}; identical raw=${JSON.stringify(restarted.frames)===original}`);
  await session.dispose();session=undefined;
  assert(document.querySelector('#preview')!.childElementCount===0,'Preview leaked after disposal');
  let rejected=false;try{const unexpected=await new BrowserVideoSource(document.querySelector<HTMLElement>('#preview')!).open(new File(['invalid'],'invalid.mp4',{type:'video/mp4'}),signal);await unexpected.dispose()}catch{rejected=true}
  assert(rejected&&document.querySelector('#preview')!.childElementCount===0,'Invalid input cleanup failed');
  log('PASS session disposal removes preview; malformed MP4 rejected without preview residue');
  outputUrl=URL.createObjectURL(new Blob([text],{type:'application/json'}));download.disabled=false;download.onclick=()=>{const a=document.createElement('a');a.href=outputUrl!;a.download='coords-1-dev-qa.json';a.click()};
  log('ALL CORE CHECKS PASSED — manual standing validation and Galaxy device checks deferred; no real server acceptance');
 }catch(e){log(`FAIL ${e instanceof Error?e.stack:e}`)}finally{await engine?.dispose();await session?.dispose();button.disabled=false}
};
