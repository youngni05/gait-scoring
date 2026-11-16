import {useEffect,useRef,useState}from'react';
import type {PoseFrame}from'../../contracts/coords-1/types';
import type {ProcessingInput,VideoInfo}from'./pose/ports';
import {JsonCoordsSerializer}from'./pose/coords-serializer';
import {MockUploadGateway,type MockScenario,type UploadRequest,validateUploadRequest}from'./pose/upload-gateway';
import {FrontalStandDetector}from'./pose/stand-detector';
export function PayloadTrial({frames,video,input}:{frames:readonly PoseFrame[];video:VideoInfo;input:ProcessingInput}){
 const [prepared,setPrepared]=useState<{request:UploadRequest;text:string;bytes:number}|null>(null);
 const [status,setStatus]=useState('아래 시험 입력을 확인한 뒤 JSON을 생성하세요. 실제 서버 연결은 없습니다.');
 const [scenario,setScenario]=useState<MockScenario>('success'),[busy,setBusy]=useState(false);
 const mode=useRef<MockScenario>('success');mode.current=scenario;
 const gateway=useRef<MockUploadGateway|null>(null);gateway.current??=new MockUploadGateway(()=>mode.current);
 const controller=useRef<AbortController|null>(null),epoch=useRef(0),url=useRef<string|null>(null);
 const inputKey=JSON.stringify(input);
 useEffect(()=>{epoch.current++;controller.current?.abort();setPrepared(null);setBusy(false);setStatus('입력값 또는 추출 결과 변경됨 · JSON을 다시 생성하세요.');if(url.current){URL.revokeObjectURL(url.current);url.current=null;}return()=>{epoch.current++;controller.current?.abort();if(url.current){URL.revokeObjectURL(url.current);url.current=null;}}},[frames,inputKey]);
 function prepare(){
  epoch.current++;controller.current?.abort();setBusy(false);setPrepared(null);if(url.current){URL.revokeObjectURL(url.current);url.current=null;}
  try{
   const detection=new FrontalStandDetector(video.width/video.height).analyze(frames,video.durationSeconds,new AbortController().signal).detection;
   const serializer=new JsonCoordsSerializer();const result=serializer.create({input,video,frames,marks:detection.marks,pose:{model:'pose_landmarker_full',engine_version:'1.2.3',delegate:'CPU'}});
   if(!result.ok)throw Error(result.issues.map(i=>`${i.path}: ${i.message}`).join('; '));
   const text=serializer.stringify(result.value),coords=JSON.parse(text);
   const request:UploadRequest={coords,requestId:crypto.randomUUID(),metadata:{recordedAtSource:input.recordedAtSource,rotationAppliedDegrees:video.rotationAppliedDegrees,detection,modelAssetVersion:'pose_landmarker_full/float16/1'}};
   validateUploadRequest(request);setPrepared({request,text,bytes:new TextEncoder().encode(text).length});setStatus(`좌표 JSON 생성·계약 검증 완료 · ${frames.length}프레임. 전송 미실행.`);
  }catch(e){setStatus(`생성 실패: ${e instanceof Error?e.message:e}`)}
 }
 async function submit(){if(!prepared||busy)return;const id=++epoch.current,abort=new AbortController();controller.current=abort;setBusy(true);setStatus('mock 전달 대기 중… 실제 서버 요청은 없습니다.');
  try{validateUploadRequest(prepared.request);const receipt=await gateway.current!.submit(prepared.request,abort.signal);if(id===epoch.current&&!abort.signal.aborted)setStatus(`mock 접수 성공 · ${receipt.receiptId}. 실제 서버 접수·분석 완료가 아닙니다.`)}
  catch(e){if(id===epoch.current)setStatus(abort.signal.aborted?'mock 전달 취소됨. 같은 ID로 다시 시도할 수 있습니다.':`mock 전달 실패: ${e instanceof Error?e.message:e}. 같은 ID로 재시도할 수 있습니다.`)}
  finally{if(id===epoch.current)setBusy(false)}
 }
 function download(){if(!prepared)return;validateUploadRequest(prepared.request);if(url.current)URL.revokeObjectURL(url.current);url.current=URL.createObjectURL(new Blob([prepared.text],{type:'application/json'}));const a=document.createElement('a');a.href=url.current;a.download=`coords-1-dev-${prepared.request.coords.recording_id}.json`;a.click();}
 return<section style={{marginTop:24}} aria-label="JSON 생성과 mock 전달"><h2>coords-1 JSON · mock 전달</h2><p className="muted">개발 시험 전용 · 실제 인증/DB/서버/분석 없음. capture.mode는 시험 계약의 guided이며 촬영 규칙 준수 확인은 아닙니다.</p>
 <button disabled={busy} onClick={prepare}>JSON 생성·검증</button><p role="status" className="status">{status}</p>
 {prepared&&<><p>본문 {prepared.bytes.toLocaleString()} bytes · {prepared.request.coords.frames.length}프레임 · 미검출 {prepared.request.coords.frames.filter(f=>f.lm===null).length}<br/>서 있기 시작 {String(prepared.request.coords.marks.stand_start)} / 끝 {String(prepared.request.coords.marks.stand_end)}</p>
 <small>기록 ID: {prepared.request.coords.recording_id}<br/>요청 ID: {prepared.request.requestId}<br/>재시도는 두 ID와 본문을 유지합니다. 입력 변경·새 JSON 생성은 별도 요청이며 실제 서버 중복 방지는 서버가 보장해야 합니다.</small>
 <details><summary>본문 정보와 별도 메타데이터 확인</summary><pre style={{fontSize:11,overflow:'auto'}}>{JSON.stringify({subject:prepared.request.coords.subject,capture:prepared.request.coords.capture,pose:prepared.request.coords.pose,metadata:prepared.request.metadata},null,2)}</pre></details>
 <button className="secondary" onClick={download}>개발용 JSON 다운로드</button><small>실제 좌표 JSON은 개인정보가 될 수 있습니다. Git에 넣지 마세요. 영상·음성·이미지·파일 경로는 본문에 포함하지 않습니다.</small>
 <label>mock 시험 상황<select aria-label="mock 시험 상황" value={scenario} onChange={e=>setScenario(e.target.value as MockScenario)} disabled={busy}><option value="success">성공</option><option value="timeout">시간 초과</option><option value="network">네트워크 실패</option><option value="auth">인증 오류</option></select></label>
 <div className="actions"><button disabled={busy} onClick={()=>void submit()}>mock 전달 / 재시도</button><button className="secondary" disabled={!busy} onClick={()=>controller.current?.abort()}>mock 전달 취소</button></div></>}
 <small>서 있기 수동 확인·갤럭시 실기기 검증은 보류 중입니다. 미검출 marks는 null로 유지합니다.</small></section>;
}
