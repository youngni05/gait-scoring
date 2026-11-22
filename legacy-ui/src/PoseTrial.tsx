import { PayloadTrial } from './PayloadTrial';
import type { ProcessingInput } from '../../mobile/src/pose/ports';
import { StandTrial } from './StandTrial';
import { useEffect, useRef, useState } from 'react';
import type { PoseRun } from '../../mobile/src/pose/extract-video';
import { extractVideo } from '../../mobile/src/pose/extract-video';
import { BrowserVideoSource, type BrowserVideoSession } from '../../mobile/src/pose/browser-video-source';
import { MediaPipePoseExtractor } from '../../mobile/src/pose/mediapipe-extractor';

export function PoseTrial({ file, input }: { file: File | null; input: ProcessingInput }) {
 const mount = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null);
 const session = useRef<BrowserVideoSession | null>(null), control = useRef<AbortController | null>(null);
 const generation = useRef(0);
 const [busy, setBusy] = useState(false), [status, setStatus] = useState('영상을 선택하고 관절 추출을 시작하세요.');
 const [result, setResult] = useState<PoseRun | null>(null), [index, setIndex] = useState(0);
 const [overlay, setOverlay] = useState(true), [view, setView] = useState('');
 const [multi, setMulti] = useState('다인 후보 점검 미실행. 한 명 출력은 단일 인물 확인이 아닙니다.');
 useEffect(() => {
  generation.current++; control.current?.abort(); void session.current?.dispose(); session.current = null;
  setBusy(false); setResult(null); setIndex(0); setView(''); setStatus('영상을 선택하고 관절 추출을 시작하세요.');
  setMulti('다인 후보 점검 미실행. 한 명 출력은 단일 인물 확인이 아닙니다.');
  return () => { generation.current++; control.current?.abort(); void session.current?.dispose(); };
 }, [file]);
 useEffect(() => {
  const frame = result?.frames[index], active = session.current, target = canvas.current;
  if (!frame || !active || !target) return;
  const abort = new AbortController(); setView('선택 프레임 디코딩 중…');
  // Draw on a private scratch buffer: an older async request cannot overwrite a new selection.
  const scratch = document.createElement('canvas');
  void (async () => {
   try {
    await active.drawFrame(frame.t, scratch, abort.signal);
    if (abort.signal.aborted) return;
    const ctx = scratch.getContext('2d')!;
    if (overlay && frame.lm) {
     const { PoseLandmarker } = await import('@mediapipe/tasks-vision');
     if (abort.signal.aborted) return;
     ctx.lineWidth = Math.max(2, scratch.width / 250); ctx.strokeStyle = '#00ff88';
     for (const edge of PoseLandmarker.POSE_CONNECTIONS) {
      const a = frame.lm[edge.start]!, b = frame.lm[edge.end]!;
      if (a[3] < .5 || b[3] < .5) continue;
      ctx.beginPath(); ctx.moveTo(a[0]*scratch.width,a[1]*scratch.height); ctx.lineTo(b[0]*scratch.width,b[1]*scratch.height); ctx.stroke();
     }
     for (let n=0;n<frame.lm.length;n++) { const p=frame.lm[n]!; if(p[3]<.5)continue;
      ctx.fillStyle = n % 2 ? '#ffcf40' : '#ff6699'; ctx.beginPath(); ctx.arc(p[0]*scratch.width,p[1]*scratch.height,Math.max(3,scratch.width/160),0,Math.PI*2); ctx.fill();
     }
    }
    target.width=scratch.width; target.height=scratch.height; target.getContext('2d')!.drawImage(scratch,0,0);
    setView(`프레임 ${index+1}/${result.frames.length} · 원본 시각 ${frame.t.toFixed(6)}초 · ${frame.lm ? '33관절' : '미검출 (lm:null)'}`);
   } catch(error) { if(!abort.signal.aborted)setView(`오류: ${error instanceof Error ? error.message : error}`); }
   finally { scratch.width=0; scratch.height=0; }
  })();
  return () => abort.abort();
 }, [result,index,overlay]);
 async function run() {
  if (!file || busy) return;
  const id=++generation.current, abort=new AbortController(); control.current=abort;
  await session.current?.dispose(); session.current=null;
  setBusy(true);setResult(null);setIndex(0);setView('');setMulti('다인 후보 점검 미실행');setStatus('로컬 full 모델·CPU 초기화 중…');
  let engine: MediaPipePoseExtractor | undefined;
  try {
   const opened=await new BrowserVideoSource(mount.current!).open(file,abort.signal);
   if(id!==generation.current){await opened.dispose();return;} session.current=opened;
   engine=await MediaPipePoseExtractor.create(abort.signal);
   const output=await extractVideo(opened,engine,abort.signal,(count,detected)=>{
    if(id===generation.current)setStatus(`관절 추출 ${count}/${opened.verification?.expected.length ?? '?'} · 검출 ${detected} · 미검출 ${count-detected}`);
   });
   if(!opened.verification?.complete || output.frames.length!==opened.verification.expected.length)throw new Error('전체 프레임 검증 미완료');
   if(id===generation.current){setResult(output);setStatus(`관절 추출 완료 · ${output.frames.length}/${opened.verification.expected.length}프레임 · 검출 ${output.detected} · 미검출 ${output.missing} · ${(output.elapsedMs/1000).toFixed(2)}초 (초기화 제외)`);}
  } catch(error) { if(id===generation.current)setStatus(abort.signal.aborted?'관절 추출 취소됨. 부분 결과를 폐기했습니다. 다시 시작할 수 있습니다.':`오류: ${error instanceof Error ? error.message : error}`); }
  finally {await engine?.dispose();if(id===generation.current)setBusy(false);}
 }
 async function inspectPeople() {
  if(!session.current || !result || busy)return;
  const active=session.current,id=++generation.current,abort=new AbortController();control.current=abort;setBusy(true);setMulti('별도 2인 후보 모델 초기화 중…');
  let engine: MediaPipePoseExtractor | undefined;
  try {
   engine=await MediaPipePoseExtractor.create(abort.signal,2);let count=0,multiple=0,max=0;
   for await(const frame of active.frames(abort.signal)) {
    try { const n=engine.countCandidates(frame,abort.signal);count++;max=Math.max(max,n);if(n>1)multiple++;
     if(id===generation.current)setMulti(`다인 후보 점검 ${count}/${result.frames.length} · 2인 후보 ${multiple}프레임`);
    } finally{frame.release();}
    await new Promise(r=>setTimeout(r,0));
   }
   if(abort.signal.aborted)throw new DOMException('취소됨','AbortError');
   if(!active.verification?.complete||count!==result.frames.length)throw new Error('다인 후보 전체 프레임 검사 미완료');
   if(id===generation.current)setMulti(`다인 후보 점검 완료 ${count}/${result.frames.length} · 최대 ${max}명 · 2인 후보 ${multiple}프레임. ${multiple?'단일 대상 영상으로 확정하지 마세요.':'미검출된 다른 사람이 없다는 보장은 아닙니다.'}`);
  }catch(error){if(id===generation.current)setMulti(abort.signal.aborted?'다인 후보 점검 취소됨 (미완료)':`오류: ${error instanceof Error?error.message:error}`);}
  finally{await engine?.dispose();if(id===generation.current)setBusy(false);}
 }
 return <section className="card">
  <h2>33관절 추출 · 프레임 탐색</h2><p className="muted">full 모델 · CPU · VIDEO 모드 · 원본 시각. 좌표는 메모리에만 유지합니다.</p>
  <div hidden ref={mount}/><div className="actions"><button disabled={!file||busy} onClick={()=>void run()}>관절 추출 시작</button><button className="secondary" disabled={!busy} onClick={()=>control.current?.abort()}>관절 처리 취소</button></div>
  <p role="status" className="status">{status}</p>
  {result && <>
   <canvas ref={canvas} aria-label="선택 프레임 관절 오버레이" style={{width:'100%',maxHeight:560,objectFit:'contain',background:'#16221c'}}/>
   <p role="status">{view}</p>
   <label>프레임 번호<input type="number" min={1} max={result.frames.length} value={index+1} onChange={e=>setIndex(Math.max(0,Math.min(result.frames.length-1,Math.trunc(Number(e.target.value))-1||0)))}/></label>
   <div className="actions"><button className="secondary" disabled={index===0} onClick={()=>setIndex(i=>i-1)}>이전 프레임</button><button className="secondary" disabled={index===result.frames.length-1} onClick={()=>setIndex(i=>i+1)}>다음 프레임</button><button className="secondary" onClick={()=>setOverlay(v=>!v)}>{overlay?'오버레이 숨기기':'오버레이 보기'}</button></div>
   <small>표시는 visibility ≥ 0.5 관절만 연결합니다. 원본 33관절 값은 필터·보간·반전 없이 보존합니다. z는 미터가 아닌 상대 깊이입니다.</small>
   <details><summary>선택 프레임 원본 33관절 값</summary><pre style={{maxHeight:220,overflow:'auto',fontSize:11}}>{JSON.stringify(result.frames[index],null,2)}</pre></details>
   {session.current && <StandTrial frames={result.frames} duration={session.current.info.durationSeconds} aspectRatio={session.current.info.width/session.current.info.height} onSeek={t=>setIndex(result.frames.reduce((best,f,i)=>Math.abs(f.t-t)<Math.abs(result.frames[best]!.t-t)?i:best,0))}/>}
   {session.current && <PayloadTrial key={JSON.stringify(input)} frames={result.frames} video={session.current.info} input={input}/>}
   <button className="secondary" disabled={busy} onClick={()=>void inspectPeople()}>다인 후보 점검</button>
  </>}
  <p className="status">{multi}</p><small>관절 정렬 확인은 임상 정확도·GDI 호환성 검증이 아닙니다. 서 있기 검출은 실험 규칙이며 실제 서버 전송은 연결 대기입니다.</small>
 </section>;
}
