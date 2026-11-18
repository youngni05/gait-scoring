import { BrowserVideoSource } from './pose/browser-video-source';
import { MediaPipePoseExtractor } from './pose/mediapipe-extractor';
import { extractVideo } from './pose/extract-video';
const assert=(ok:unknown,message:string)=>{if(!ok)throw Error(message)};
document.querySelector<HTMLButtonElement>('#run')!.onclick=async()=>{
 const out=document.querySelector('#result')!,button=document.querySelector<HTMLButtonElement>('#run')!;button.disabled=true;out.textContent='';
 const log=(s:string)=>{out.textContent+=s+'\n'};
 const signal=new AbortController().signal;
 let session:Awaited<ReturnType<BrowserVideoSource['open']>>|undefined, engine:MediaPipePoseExtractor|undefined;
 const requests:{method:string;local:boolean;body:boolean;rejected?:boolean}[]=[];
 const blocked:string[]=[];const onViolation=(e:SecurityPolicyViolationEvent)=>{if(e.effectiveDirective==='connect-src')blocked.push(e.blockedURI)};document.addEventListener('securitypolicyviolation',onViolation);
 const oldFetch=window.fetch;
 const oldOpen=XMLHttpRequest.prototype.open,oldSend=XMLHttpRequest.prototype.send;
 const xhrInfo=new WeakMap<XMLHttpRequest,{method:string;local:boolean;body:boolean}>();
 XMLHttpRequest.prototype.open=function(method:string,url:string|URL,async:boolean=true,username?:string|null,password?:string|null){xhrInfo.set(this,{method:method.toUpperCase(),local:new URL(String(url),location.href).origin===location.origin,body:false});return oldOpen.call(this,method,url,async,username,password)};
 XMLHttpRequest.prototype.send=function(body?:Document|XMLHttpRequestBodyInit|null){const entry=xhrInfo.get(this);if(entry){entry.body=body!=null;requests.push(entry)}return oldSend.call(this,body)};
 window.fetch=async(input,init)=>{const url=input instanceof Request?input.url:String(input);const entry={method:init?.method??(input instanceof Request?input.method:'GET'),local:new URL(url,location.href).origin===location.origin,body:!!init?.body||(input instanceof Request&&input.body!==null),rejected:false};requests.push(entry);try{return await oldFetch(input,init)}catch(e){entry.rejected=true;throw e}};

 try{
  const files=[...document.querySelector<HTMLInputElement>('#files')!.files!];
  const expected:number[]=JSON.parse(await files.find(f=>f.name.endsWith('.json'))!.text());
  session=await new BrowserVideoSource(document.querySelector<HTMLElement>('#preview')!).open(files.find(f=>f.name.endsWith('.mp4'))!,signal);
  engine=await MediaPipePoseExtractor.create(signal);
  const result=await extractVideo(session,engine,signal,(n)=>{if(n%50===0)log(`processing ${n}/${expected.length}`)});
  assert(session.verification?.complete&&result.frames.length===expected.length,'Incomplete decoding');
  result.frames.forEach((frame,i)=>{
   assert(Math.abs(frame.t-expected[i]!)<1e-8,'Original PTS mismatch');
   if(frame.lm)assert(frame.lm.length===33&&frame.lm.every(p=>p.length===4&&p.every(Number.isFinite)),'Invalid landmark values');
  });
  log(`PASS CPU/full VIDEO ${result.frames.length}/${expected.length}; detected=${result.detected}; null=${result.missing}; elapsed=${(result.elapsedMs/1000).toFixed(2)}s; ALL PTS matched`);
  log('Null frame numbers: '+result.frames.flatMap((f,i)=>f.lm?[]:[i+1]).join(','));
  await engine.dispose(); engine=undefined;
  const canvas=document.createElement('canvas');
  for(const i of [0,Math.floor(expected.length/2),expected.length-1]){await session.drawFrame(expected[i]!,canvas,signal);assert(canvas.width===session.info.width&&canvas.height===session.info.height,'Inspection size');}
  log('PASS exact-frame inspection at first/middle/last');
  // Separate diagnostic model, not the single-person extractor. Test blank and duplicated source frame.
  engine=await MediaPipePoseExtractor.create(signal,2);
  const blank=document.createElement('canvas');blank.width=256;blank.height=256;blank.getContext('2d')!.fillRect(0,0,256,256);
  assert(engine.countCandidates({t:0,pixels:blank,release(){}},signal)===0,'Blank candidate hallucination');
  await session.drawFrame(expected[Math.floor(expected.length/2)]!,canvas,signal);
  const collage=document.createElement('canvas');collage.width=canvas.width*2;collage.height=canvas.height;
  collage.getContext('2d')!.drawImage(canvas,0,0);collage.getContext('2d')!.drawImage(canvas,canvas.width,0);
  const candidates=engine.countCandidates({t:1,pixels:collage,release(){}},signal);
  log(`DIAGNOSTIC duplicated-person canvas: ${candidates} candidates (expected 2; real multi-person accuracy remains unverified)`);
  await engine.dispose(); engine=await MediaPipePoseExtractor.create(signal,2);
  let count=0,multi=0,max=0;const candidateFrames:number[]=[];
  for await(const frame of session.frames(signal)){try{const n=engine.countCandidates(frame,signal);count++;if(count%100===0)log(`candidate scan ${count}/${expected.length}`);max=Math.max(max,n);if(n>1){multi++;candidateFrames.push(count);}}finally{frame.release()}await new Promise(r=>setTimeout(r,0))}
  assert(count===expected.length&&session.verification?.complete,'Incomplete candidate inspection');
  log(`PASS separate candidate scan ${count}/${expected.length}; max=${max}; multiple=${multi} frames (not proof of one person)`);
  log('Multiple-candidate frame numbers: '+candidateFrames.join(','));
  await engine.dispose();engine=await MediaPipePoseExtractor.create(signal);
  assert((await engine.extract({t:0,pixels:blank,release(){}},signal)).lm===null,'Blank must be null');
  await engine.dispose();engine=await MediaPipePoseExtractor.create(signal);
  const abort=new AbortController();let cancelled=false;
  try{await extractVideo(session,engine,abort.signal,n=>{if(n===17)abort.abort()})}catch(e){cancelled=e instanceof DOMException&&e.name==='AbortError'}
  assert(cancelled&&!session.verification?.complete,'Cancellation did not remain incomplete');
  await engine.dispose();engine=await MediaPipePoseExtractor.create(signal);
  const restart=await extractVideo(session,engine,signal,n=>{if(n%100===0)log(`restart ${n}/${expected.length}`)});
  assert(restart.frames.length===expected.length&&session.verification?.complete,'Restart incomplete');
  restart.frames.forEach((f,i)=>assert(f.t===result.frames[i]!.t,'Restart PTS mismatch'));
  log(`PASS blank lm:null; cancel at17; restart ${restart.frames.length}/${expected.length} detected=${restart.detected} null=${restart.missing}`);
  await engine.dispose();engine=undefined;await new Promise(r=>setTimeout(r,100));
  assert(requests.every(r=>(r.local&&r.method==='GET'&&!r.body)||(!r.local&&r.rejected)),'Unexpected successful request');
  assert(blocked.length>0&&blocked.every(url=>url.startsWith('https://odml.pa.googleapis.com/')),'Expected CSP telemetry blocking not observed');
  log(`PASS fetch/XHR audit: local GET=${requests.filter(r=>r.local).length}; external attempts rejected=${requests.filter(r=>!r.local&&r.rejected).length}; CSP connect-src violations=${blocked.length}; no successful external request`);
  canvas.width=0;blank.width=0;collage.width=0;
  log('CORE CHECKS PASSED; multi-person accuracy remains unverified (see diagnostic above)');
 }catch(e){log(`FAIL ${e instanceof Error?e.stack:e}`)}
 finally{await engine?.dispose();await session?.dispose();document.removeEventListener('securitypolicyviolation',onViolation);window.fetch=oldFetch;XMLHttpRequest.prototype.open=oldOpen;XMLHttpRequest.prototype.send=oldSend;button.disabled=false}
};
