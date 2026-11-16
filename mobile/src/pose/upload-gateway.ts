import { validateCoords } from '../../../contracts/coords-1/validate';
import type { UploadGateway } from './ports';
export type UploadRequest=Parameters<UploadGateway['submit']>[0];
export type MockScenario='success'|'timeout'|'network'|'auth';
export class UploadFailure extends Error {constructor(readonly code:'timeout'|'network'|'auth'|'conflict',message:string){super(message);this.name='UploadFailure';}}
export function validateUploadRequest(request:UploadRequest):void {
 const result=validateCoords(request.coords);if(!result.ok)throw Error('전달 직전 coords-1 검증 실패: '+result.issues.map(i=>i.message).join('; '));
 if(Object.keys(request).some(k=>!['coords','metadata','requestId'].includes(k))||!request.requestId.trim())throw Error('비정상 전달 요청');
 const m=request.metadata;
 if(Object.keys(m).some(k=>!['recordedAtSource','rotationAppliedDegrees','detection','modelAssetVersion'].includes(k)))throw Error('허용하지 않은 메타데이터 필드');
 if(!['user_entered','metadata'].includes(m.recordedAtSource)||!(m.rotationAppliedDegrees===null||Number.isFinite(m.rotationAppliedDegrees)))throw Error('비정상 메타데이터');
 const d=m.detection;
 if(Object.keys(d).some(k=>!['status','marks','version','reason'].includes(k))||typeof d.version!=='string'||!d.version||!['detected','not_detected'].includes(d.status))throw Error('비정상 검출 메타데이터');
 if(d.status==='not_detected'&&(typeof d.reason!=='string'||d.marks.stand_start!==null||d.marks.stand_end!==null))throw Error('미검출 메타데이터 오류');
 if(d.status==='detected'&&(d.marks.stand_start===null||d.marks.stand_end===null))throw Error('검출 메타데이터 오류');
 if(JSON.stringify(d.marks)!==JSON.stringify(request.coords.marks))throw Error('본문과 메타데이터의 검출 구간 불일치');
 if(m.modelAssetVersion!==undefined&&typeof m.modelAssetVersion!=='string')throw Error('비정상 모델 버전');
}
function wait(ms:number,signal:AbortSignal):Promise<void>{return new Promise((resolve,reject)=>{const abort=()=>{clearTimeout(timer);signal.removeEventListener('abort',abort);reject(new DOMException('mock 전달 취소','AbortError'))};const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve()},ms);signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();});}
/** No fetch/XHR, tokens, persistence or server. Receipts exist only for this mock instance. */
export class MockUploadGateway implements UploadGateway {
 private receipts=new Map<string,{body:string;receiptId:string}>();
 constructor(private readonly scenario:()=>MockScenario,private delayMs=1000){}
 async submit(request:UploadRequest,signal:AbortSignal):Promise<{receiptId:string;environment:'mock'}>{
  validateUploadRequest(request);const body=JSON.stringify(request);const snapshot:UploadRequest=JSON.parse(body);const mode=this.scenario();
  await wait(this.delayMs,signal);if(signal.aborted)throw new DOMException('mock 전달 취소','AbortError');
  const previous=this.receipts.get(snapshot.requestId);
  if(previous&&previous.body!==body)throw new UploadFailure('conflict','같은 요청 ID에 다른 본문이 연결됐습니다.');
  if(mode==='timeout')throw new UploadFailure('timeout','mock 시간 초과 (실제 서버 요청 없음)');
  if(mode==='network')throw new UploadFailure('network','mock 네트워크 실패 (실제 네트워크 사용 없음)');
  if(mode==='auth')throw new UploadFailure('auth','mock 인증 오류 (실제 인증 수행 안 함)');
  if(previous)return{receiptId:previous.receiptId,environment:'mock'};
  const receiptId=`mock-${snapshot.requestId}`;this.receipts.set(snapshot.requestId,{body,receiptId});return{receiptId,environment:'mock'};
 }
}
/** Integration seam only. An agreed HTTP/auth contract must supply this transport; UI never instantiates it. */
export class InjectedUploadGateway implements UploadGateway {
 constructor(private transport:UploadGateway['submit']){}
 async submit(request:UploadRequest,signal:AbortSignal){validateUploadRequest(request);if(signal.aborted)throw new DOMException('취소됨','AbortError');return this.transport(JSON.parse(JSON.stringify(request)),signal);}
}
